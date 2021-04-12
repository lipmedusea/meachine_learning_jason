from data_treatment import data_clean2
from data_treatment import load_data_new
import joblib
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':
    sql = " "
    df = load_data_new(sql, filename="btest.csv")
    df = data_clean2(df)
    labels = "target_is_DD_ACTIVE"
    select_columns = [
        'is_festival_user',
        'is_LAST_2YEAR_DD_ACTIVE',
        'cafe_tag_is_mop_available',
        'IS_SR_KIT_USER',

        'level_use',
        'skr_rate',
        'merch_rate',
        'active_index',
        'cafe_tag_p6m_food_qty',
        'DD_rev',
        'svc_revenue',
        'SR_KIT_NUM',
        'cafe_tag_p3m_merch_party_size',
        'CAFE20_VISIT_MERCH',
        'CAFE20_AMT',
        'cafe_tag_p3m_food_qty',
        'p3m_weekday_trans',
        'max_DD_rev',
        'DD_end_gap',
        'MC_end_gap',
        'p2w_amt',
        'cafe_tag_p6m_merch_qty',
        'MCoffer_red',
        'p2w_trans',
        'CAFE20_RECENCY_SRKIT',
        labels,
    ]
    catfeatures = [
        'is_festival_user',
        'is_LAST_2YEAR_DD_ACTIVE',
        'cafe_tag_is_mop_available',
        'IS_SR_KIT_USER']
    df = df[select_columns]
    for cats in catfeatures:
        df[cats] = df[cats].astype(int)

    x = df.drop(labels, axis=1)
    y = df[labels]

    adaboost = joblib.load("models/adaboost.model")
    lr = joblib.load("models/lr.model")
    rf = joblib.load("models/rf.model")
    gbdt = joblib.load("models/gbdt.model")
    xgboost = joblib.load("models/xgboost.model")
    lgb = joblib.load("models/lgb.model")
    catboost = joblib.load("models/catboost.model")

    prediction_adaboost = pd.DataFrame(adaboost.predict_proba(x)[:, 1])
    prediction_lr = pd.DataFrame(lr.predict_proba(x)[:, 1])
    prediction_rf = pd.DataFrame(rf.predict_proba(x)[:, 1])
    prediction_gbdt = pd.DataFrame(gbdt.predict_proba(x)[:, 1])
    prediction_xgboost = pd.DataFrame(xgboost.predict_proba(x)[:, 1])
    prediction_lgb = pd.DataFrame(lgb.predict(x))
    prediction_catboost = pd.DataFrame(catboost.predict_proba(x)[:, 1])

    prediction_adaboost["adaboost_rank"] = prediction_adaboost.rank(ascending=False, method="min")
    prediction_lr["lr_rank"] = prediction_lr.rank(ascending=False, method="min")
    prediction_rf["rf_rank"] = prediction_rf.rank(ascending=False, method="min")
    prediction_gbdt["gbdt_rank"] = prediction_gbdt.rank(ascending=False, method="min")
    prediction_xgboost["xgboost_rank"] = prediction_xgboost.rank(ascending=False, method="min")
    prediction_lgb["lgb_rank"] = prediction_lgb.rank(ascending=False, method="min")
    prediction_catboost["adaboost_rank"] = prediction_catboost.rank(ascending=False, method="min")

    df_rank = pd.concat([
        # prediction_adaboost,
        # prediction_lr,
        # prediction_rf,
        prediction_gbdt,
        prediction_xgboost,
        prediction_lgb,
        # prediction_catboost
    ],
    axis=1)

    df_rank.drop(0, axis=1, inplace=True)
    df_rank["min"] = df_rank.min(axis=1)
    df_rank["max"] = df_rank.max(axis=1)

    df_rank["diffs"] = df_rank["max"] - df_rank["min"]

    df_rank["diffs"].hist(bins=range(0, 26000, 500),
                          # 20,
                          # [0, 500, 1000, 1500, 2000, 2500, 3000,
                          #  3500, 4000, 4500, 5000, 5500, 6000,
                          #  5000, 10000, 26000],
                         facecolor='blue',
                         alpha=0.5,
                         edgecolor='red')
    plt.show()

    distribution = df_rank.groupby(["diffs"])['diffs'].count()
    print(df_rank["diffs"].max())





