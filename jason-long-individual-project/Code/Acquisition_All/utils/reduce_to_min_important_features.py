def to_important_features(std_dev_mult=2):
    def reduce_to_min_important_features(agg, imp, clf):
        top_imp = imp[imp >= imp.std() * std_dev_mult]
        if not agg:
            print('Choosing top importances', clf)
            return top_imp.to_dict()
        if len(top_imp) < len(agg):
            print('Switching top importances', clf)
            return top_imp.to_dict()
        return agg

    return reduce_to_min_important_features
