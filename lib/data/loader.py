import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader
import pandas as pd
from numpy import vectorize
import re

from lib.db import connect
from lib.helpers.pre_process import PreProcessors
from lib.enums import PRE_PROCESSING_ENCODERS_PICKLE_PATH, LIVE_PRE_PROCESSING_ENCODERS_PICKLE_PATH


# https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
# https://loanperformancedata.fanniemae.com/lppub/index.html
# https://console.cloud.google.com/compute/instances?authuser=3&project=polar-protocol-228721&instancessize=50&duration=PT1H
# http://initd.org/psycopg/docs/usage.html

@vectorize
def make_snake(name, column):
    n = re.sub('[^A-Za-z0-9 ]+', '', str(name)).lower().replace(' ', '_')
    return '{}_{}'.format(column, n)


class LoanPerformanceDataset(TorchDataset):
    def __init__(self,
                 chunk=1000,
                 conn=None,
                 split_ratio=None,
                 headers=None,
                 ignore_headers=None,
                 target_column=None,
                 pre_process_pickle_path=None,
                 stage=None,
                 to_tensor=True
                 ):
        self.conn = conn
        self.headers = [h for h in (headers or self._default_headers) if h not in (ignore_headers or [])]
        self.target_column = target_column
        self.split_ratio = split_ratio or [78, 11, 11]
        self.stage = stage or 'train'
        self.chunk = chunk
        self.to_tensor = to_tensor
        self.len_ = 0
        self.pre_process_pickle_path = pre_process_pickle_path
        self.proxy = self._set_proxy()
        self.target_proxy = self._set_target_proxy()
        self.cat_encoders, self.acq_num_encoder, self.perf_num_encoder, self.target_encoder = self._load_encoders()

    @property
    def _default_headers(self):
        return self.conn.execute(
            "SELECT * FROM acquisition LIMIT 1"
        ).keys()

    def _set_proxy(self):
        q = "SELECT {} FROM {}".format(
            ','.join(self.headers) or '*',
            self.stage,
        )
        return self.conn.execution_options(stream_results=True).execute(q)

    def _set_target_proxy(self):
        q = """
              SELECT {} FROM {} 
              WHERE ABS(MOD(loan_id, 8)) = 2
              AND sdq > 0
              """.format(
            ','.join(self.headers) or '*',
            self.stage,
        )
        return self.conn.execution_options(stream_results=True).execute(q)

    def _iterate(self, proxy, chunk=None):
        while True:
            batch = proxy.fetchmany(chunk or self.chunk)
            if not batch:
                proxy.close()
                return
            yield batch

    def _load_encoders(self):
        pp = PreProcessors()
        pp.load(self.pre_process_pickle_path)

        acq = pp.encoders.get('acquisition_numerical')
        acq_cat = pp.encoders.get('acquisition_categorical')
        tgt = pp.encoders.get('target')

        cat_encoders = dict()
        cat_encoders.update(acq_cat.encoder.__dict__)
        return cat_encoders, acq, None, tgt

    def _encode(self, df):
        categories = df.select_dtypes(include='object')
        acq_standard = self.acq_num_encoder.encoder.standard.transform(
            df[self.acq_num_encoder.columns].astype('float64')
        )

        frames = [
            pd.DataFrame(acq_standard, columns=self.acq_num_encoder.columns).fillna(0),
        ]

        for column in categories:
            if column != self.target_column:
                encoder = self.cat_encoders.get(column)
                if encoder:
                    try:
                        data = df[[column]].astype(str).fillna('X') if column == 'zip_code_short' else \
                            df[[column]].fillna('X')
                        transformed = encoder.transform(data)
                        columns = make_snake(encoder.categories_, column)
                        t_df = pd.DataFrame(
                            transformed,
                            columns=columns[0]
                        )
                        frames.append(t_df)
                    except Exception as ex:
                        print(self.cat_encoders.get(column).categories_)
                        print(data.values)
                        print('ERROR IN CATEGORICAL TRANSFORM:', repr(ex))
        # targets = self.target_encoder.encoder.target.transform(df[self.target_column].values)
        return pd.concat(frames, axis=1), df[self.target_column].fillna(0)

    def __getitem__(self, index):
        nxt = next(self._iterate(self.proxy))
        nxt_tgts = next(self._iterate(self.target_proxy, int(self.chunk * .25)))
        df = pd.DataFrame(nxt + nxt_tgts, columns=self.headers).sample(frac=1)
        features, targets = self._encode(df)
        c = features.copy()
        c['sdq'] = targets.values
        c.to_csv("{}_data.csv".format(self.stage), index=False)
        if self.to_tensor:
            return torch.from_numpy(features.values).type(torch.FloatTensor), torch.tensor(targets.values,
                                                                                           dtype=torch.float)
        return features, targets

    def set_stage(self, stage):
        self.stage = stage
        self.proxy = self._set_proxy()
        self.target_proxy = self._set_target_proxy()
        return self.stage

    @property
    def len(self):
        if not self.len_:
            self.len_ = self.conn.execute("SELECT COUNT(loan_id) FROM {}".format(self.stage)).scalar()
        return self.len_

    def __len__(self):
        return self.len


"""
# curl -o acquisition.zip 'https://loanperformancedata.fanniemae.com/lppub/images/animated_favicon.gif' -H 'Referer: https://loanperformancedata.fanniemae.com/lppub/index.html' -H 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.86 Safari/537.36' -H 'DNT: 1' --compressed
curl -o performance.zip 'https://loanperformancedata.fanniemae.com/lppub/getSingleDownloadJson.json?_=1554581214775&_search=false&nd=1554581214774&rows=20&page=1&sidx=&sord=asc' -H 'Cookie: __unam=6154167-169bb7ec66a-16134105-6; TS01dc7043_28=01b40d33cfe0bc0cdeb015e955a4280799e9d52835ccab9cff57ee7aa851b3b6b7436fdeda9103c7e26e5bfae8144eb825712637b4; TS01dc7043=01fd0c5644f73ead6cae5e9846a16a0a6baafaf72401d80c33c66d3206212e2e2bf2417dc4a0767f743ffbf93991c001a1d1354ab5370d81f11c5fdb08f0e0c5b64dc9bfb0; JSESSIONID=_mb0Qr_cfajcvYyVVdDzUSs18SwvP3NV_75dF4TxGl0H5xENtO_l!484055750; TS01647536=0181ac90b091788963a2b585e66cc6926849162154ca07ef9f1610aac78d79ee23cf2fd48a10292deb80de3514747a52e5efb50f859a64266290d6e7df16eb91d37a7d48fe' -H 'DNT: 1' -H 'Accept-Encoding: gzip, deflate, br' -H 'Accept-Language: en-US,en;q=0.9' -H 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.86 Safari/537.36' -H 'Accept: application/json, text/javascript, */*' -H 'Referer: https://loanperformancedata.fanniemae.com/lppub/index.html' -H 'X-Requested-With: XMLHttpRequest' -H 'Connection: keep-alive' --compressed
curl -o performance.zip 'https://fnma-lppub-prod-us-east-1-content-private.s3.amazonaws.com/Performance_All.zip?X-Amz-Security-Token=FQoGZXIvYXdzELX%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaDJsyIEfkzqxcOB9MXyLxAsTGr7htJ3Qsi6sv4KJWPdS5SMcWwOdXMRFe9BiwNHXnu6gaXDwS4dSKPwRJgZRpcjCaVIoWi%2FnOmrB194%2Bh81SUcjvd1AYNTNlHvYvxkNbki5IaVKWQs%2BiIP8gH%2FmEkVs7U9AIF9ntx9rmyg6zIfNEV01m5d%2FTGunfepPAzBrDcpCdm%2F6aLZyEN%2F%2FE9T172taaQXOeiBSZ5xSukel393HTqYE2BC%2Fs%2BVLbN4sogWdG4BEpvEyyFF3mHKQ676dD0VY01FIzLVD2631ZueLIue2I9y0anxK2w7zCQG8tEplIEXisNiw%2Fp3MetJbd4KNComYp4F8sdpD0bA3tj4%2BKqYyl4TjOSdsyZPXHBySVrDqlisWN%2BCxImVvvO57wRMRfBA9CrJ96CxiiPk3fmS7T6Ov5kyxhHA0sHjCg1iZIozn896JAPGNH%2B%2FdHCBJhdLRtdO6%2F6teeLLksIuu5%2FqNEgc4Yoc66HlZNd9unE%2BLavNBuVtCjC4O3lBQ%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20190420T191746Z&X-Amz-SignedHeaders=host&X-Amz-Expires=59&X-Amz-Credential=ASIARRNTLK4WGNJB6Q5K%2F20190420%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=55dd48d9f66cdd21f94db4fc9095f058f84519adff03108b4aaad89142622399' -H 'Upgrade-Insecure-Requests: 1' -H 'DNT: 1' -H 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.103 Safari/537.36' -H 'Referer: https://loanperformancedata.fanniemae.com/lppub/index.html' --compressed 
curl 'https://fnma-lppub-prod-us-east-1-content-private.s3.amazonaws.com/2011Q2.zip?X-Amz-Security-Token=FQoGZXIvYXdzELX%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaDIUSsFz7QRkuPZgOPCLxAhyp8UxVW1rGL%2Fy0tiqViDB%2FWb0O3QN%2BhRwWdo9Bce8sNE1dqMXHE6YgJc5XLGmBf5%2FGv1ZnpjTUzIx5VT9kvGTag%2FCp2aWBDY%2FRtEOzHa%2BLV8yzMqQ9gkCLSOjb2HDUxhFPZKRedsU2neGOYSkh3jJ5qnWDB9JHBDEZYNfYcNYhu%2BPwLG44xYFpg%2BI2AfiA%2F0l4itF3793FT%2B7%2FQg29%2BheC6cka339liRiWbdyxa7YPE%2BSrZZTGBq2HXO54cIDvaxg%2FfuxI0vWJGfAkRIL0fFybcqp8VEI7prBBBEo3R29xBY287hCnS9%2BHzRBqg1FemwKDElHY3wP%2BGaZecaKwvZ16Blg0gxUSNBaDzqy8jO%2FDHKw5J7N8gUYfUv6p1EZ7jBuRy1L7eI4C4WKLOu6Kz%2Bu0myxsK8sg1LUBifIgc3Mn%2BIxapmw8aUsFaPbvB4mmlR%2FrlRncXCGp5dhWr%2BD2Gxn9%2F8vdB5raZXVoWaBhWAWaLijT3e3lBQ%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20190420T191124Z&X-Amz-SignedHeaders=host&X-Amz-Expires=59&X-Amz-Credential=ASIARRNTLK4WFMNGUIVB%2F20190420%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=a63cf63603f452a0f5ea070fec297ebbb7dabb90f34a028e4dff9753ee719e61' -H 'Upgrade-Insecure-Requests: 1' -H 'DNT: 1' -H 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.103 Safari/537.36' -H 'Referer: https://loanperformancedata.fanniemae.com/lppub/index.html' --compressed
curl 'https://fnma-lppub-prod-us-east-1-content-private.s3.amazonaws.com/2011Q2.zip?X-Amz-Security-Token=FQoGZXIvYXdzELX%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaDIUSsFz7QRkuPZgOPCLxAhyp8UxVW1rGL%2Fy0tiqViDB%2FWb0O3QN%2BhRwWdo9Bce8sNE1dqMXHE6YgJc5XLGmBf5%2FGv1ZnpjTUzIx5VT9kvGTag%2FCp2aWBDY%2FRtEOzHa%2BLV8yzMqQ9gkCLSOjb2HDUxhFPZKRedsU2neGOYSkh3jJ5qnWDB9JHBDEZYNfYcNYhu%2BPwLG44xYFpg%2BI2AfiA%2F0l4itF3793FT%2B7%2FQg29%2BheC6cka339liRiWbdyxa7YPE%2BSrZZTGBq2HXO54cIDvaxg%2FfuxI0vWJGfAkRIL0fFybcqp8VEI7prBBBEo3R29xBY287hCnS9%2BHzRBqg1FemwKDElHY3wP%2BGaZecaKwvZ16Blg0gxUSNBaDzqy8jO%2FDHKw5J7N8gUYfUv6p1EZ7jBuRy1L7eI4C4WKLOu6Kz%2Bu0myxsK8sg1LUBifIgc3Mn%2BIxapmw8aUsFaPbvB4mmlR%2FrlRncXCGp5dhWr%2BD2Gxn9%2F8vdB5raZXVoWaBhWAWaLijT3e3lBQ%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20190420T191124Z&X-Amz-SignedHeaders=host&X-Amz-Expires=59&X-Amz-Credential=ASIARRNTLK4WFMNGUIVB%2F20190420%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=a63cf63603f452a0f5ea070fec297ebbb7dabb90f34a028e4dff9753ee719e61' -H 'Upgrade-Insecure-Requests: 1' -H 'DNT: 1' -H 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.103 Safari/537.36' -H 'Referer: https://loanperformancedata.fanniemae.com/lppub/index.html' --compressed
"""

if __name__ == '__main__':

    def run(dataset, stage):
        loader = DataLoader(
            dataset,
            batch_size=1,  # size of batches from the query, 1 === size of query, 2 = 1/2 query size
            num_workers=1,
            shuffle=False
        )
        for batch_idx, (features, targets) in enumerate(loader):
            print('batch: {} size: {}'.format(batch_idx, targets.size()))
            print(features.size())
            # send data to model
            break  # one iteration for testing


    LOCAL = False
    pd.set_option('display.max_columns', None)  # or 1000
    dataset = LoanPerformanceDataset(
        chunk=10000,  # size of the query (use a large number here)
        conn=connect(local=LOCAL).connect(),
        ignore_headers=['loan_id'],
        target_column='sdq',
        stage='train',
        # stage='test',
        # stage='validate',
        pre_process_pickle_path='../../' + (
            PRE_PROCESSING_ENCODERS_PICKLE_PATH if LOCAL else LIVE_PRE_PROCESSING_ENCODERS_PICKLE_PATH)
    )
    for stage in ['train', 'test', 'validate']:
        dataset.set_stage(stage)
        run(dataset, stage)
