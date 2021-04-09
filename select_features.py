import warnings
from data_treatment import data_clean2, load_data_new
from data_treatment import load_data_yf,seperate_label,data_seperate
from model_evalu import evalution_model,plot_importance
import numpy as np
from models import adaboost_model, lr_model,rf_mdoel,gbdt_mdoel,xgb_model,cat_boost_model,lgb_model
import pandas as pd
import multiprocessing
from model_evalu import SBS
import lightgbm as lgb
from sklearn.feature_selection import RFECV
from collections import Counter
pd.options.mode.chained_assignment = None
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import pymysql
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from  data_treatment import load_data_yf,seperate_label, data_seperate
from model_evalu import evalution_model,plot_importance
import numpy as np
from models import rf_mdoel,gbdt_mdoel,xgb_model,cat_boost_model
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_score,f1_score,recall_score
from sklearn.base import clone
if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    from sklearn.metrics import precision_score, f1_score, recall_score

    warnings.filterwarnings('ignore')
    # 加载数据
    sql = "SELECT * from bidata.trail_pigeon_wdf1"
    df = load_data_new(sql, filename="drumping.csv")

    label_by_contract = "target_is_DD_ACTIVE"
    labels = label_by_contract

    select_columns = [
        "CAFE20_gender",
        "CAFE20_region",
        "CAFE20_levels",
        "is_festival_user",
        "level_use",
        "is_LAST_2YEAR_DD_ACTIVE",
        "cafe_tag_is_mop_available",
        "is_merch_user",
        "p4week_active",
        "is_LAST_1YEAR_DD_ACTIVE",
        "msr_lifestatus",
        "IS_SR_KIT_USER",
        "member_monetary",

        "skr_rate",
        "merch_rate",
        "bev_food_rate",
        "food_rate",
        "p6m_avg_order_amt",
        "DD_red_rate",
        "citytier",
        "active_index",
        "cafe_tag_p6m_food_qty",
        "total_amt",
        "DD_rev",
        "svc_revenue",
        "DDoffer_rec",
        "mop_spend",
        "recency",
        "food_party_size",
        "multi_bev",
        "MC_red_rate",
        "SR_KIT_NUM",
        "CAFE20_RECENCY_MERCH",
        "cafe_tag_p3m_merch_party_size",
        "CAFE20_VISIT_MERCH",
        "CAFE20_P1Y_AVG_TRANX_DAY",
        "CAFE20_VISIT_APP",
        "CAFE20_AI",
        "CAFE20_age",
        "CAFE20_RECENCY_APP",
        "CAFE20_RECENCY_bev_food",
        "CAFE20_AMT",
        "cafe_tag_p3m_food_qty",
        "rank_preference_food",
        "p3m_weekday_trans",
        "CAFE20_MONTHLY_FREQ",
        "monthly_freq",
        "cafe_tag_p6m_merch_party_size",
        "CAFE20_VISIT_bev_food",
        "max_DD_rev",
        "p6m_trans",
        "cafe_tag_p6m_monthly_freq",
        "DD_end_gap",
        "DD_launch_gap",
        "d10_p8week_active",
        "p6m_amt",
        "cafe_tag_p3m_merch_qty",
        "DD_order_num",
        "MC_end_gap",
        "p2w_amt",
        "CAFE20_VISIT_BEV",
        "CAFE20_RECENCY",
        "cafe_tag_p3m_monthly_freq",
        "cafe_tag_p6m_merch_qty",
        "p3m_weekly_frq",
        "total_trans",
        "DD_units",
        "max_DD_Quantity",
        "MC_rev",
        "p6m_weekday_trans",
        "MC_launch_gap",
        "MC_units",
        "cafe_tag_p3m_vist",
        "MCoffer_red",
        "CAFE20_VISIT_SRKIT",
        "p2w_trans",
        "CAFE20_RECENCY_SRKIT",
        "max_MC_rev",
        "CAFE20_P1Y_VISITS_DAY",
        "max_MC_Quantity",
        labels,
    ]
    catfeatures = [
        "CAFE20_gender",
        "CAFE20_region",
        "CAFE20_levels",
        "is_festival_user",
        "level_use",
        "is_LAST_2YEAR_DD_ACTIVE",
        "cafe_tag_is_mop_available",
        "is_merch_user",
        "p4week_active",
        "is_LAST_1YEAR_DD_ACTIVE",
        "msr_lifestatus",
        "IS_SR_KIT_USER",
        "member_monetary"]

    select_columns = [
        "CAFE20_gender",
        "CAFE20_region",
        "CAFE20_levels",
        "is_festival_user",
        "level_use",
        "is_LAST_2YEAR_DD_ACTIVE",
        "cafe_tag_is_mop_available",
        "is_merch_user",
        "p4week_active",
        "is_LAST_1YEAR_DD_ACTIVE",
        "msr_lifestatus",
        "IS_SR_KIT_USER",
        "member_monetary",

        "skr_rate",
        "merch_rate",
        "bev_food_rate",
        "food_rate",
        "p6m_avg_order_amt",
        "DD_red_rate",
        "citytier",
        "active_index",
        "cafe_tag_p6m_food_qty",
        "total_amt",
        "DD_rev",
        "svc_revenue",
        "DDoffer_rec",
        "mop_spend",
        "recency",
        "food_party_size",
        "multi_bev",
        "MC_red_rate",
        "SR_KIT_NUM",
        "CAFE20_RECENCY_MERCH",
        "cafe_tag_p3m_merch_party_size",
        "CAFE20_VISIT_MERCH",
        "CAFE20_P1Y_AVG_TRANX_DAY",
        "CAFE20_VISIT_APP",
        "CAFE20_AI",
        "CAFE20_age",
        "CAFE20_RECENCY_APP",
        "CAFE20_RECENCY_bev_food",
        "CAFE20_AMT",
        "cafe_tag_p3m_food_qty",
        "rank_preference_food",
        "p3m_weekday_trans",
        "CAFE20_MONTHLY_FREQ",
        "monthly_freq",
        "cafe_tag_p6m_merch_party_size",
        "CAFE20_VISIT_bev_food",
        "max_DD_rev",
        "p6m_trans",
        "cafe_tag_p6m_monthly_freq",
        "DD_end_gap",
        "DD_launch_gap",
        "d10_p8week_active",
        "p6m_amt",
        "cafe_tag_p3m_merch_qty",
        "DD_order_num",
        "MC_end_gap",
        "p2w_amt",
        "CAFE20_VISIT_BEV",
        "CAFE20_RECENCY",
        "cafe_tag_p3m_monthly_freq",
        "cafe_tag_p6m_merch_qty",
        "p3m_weekly_frq",
        "total_trans",
        "DD_units",
        "max_DD_Quantity",
        "MC_rev",
        "p6m_weekday_trans",
        "MC_launch_gap",
        "MC_units",
        "cafe_tag_p3m_vist",
        "MCoffer_red",
        "CAFE20_VISIT_SRKIT",
        "p2w_trans",
        "CAFE20_RECENCY_SRKIT",
        "max_MC_rev",
        "CAFE20_P1Y_VISITS_DAY",
        "max_MC_Quantity",
        labels,
    ]
    catfeatures = [
        "CAFE20_gender",
        "CAFE20_region",
        "CAFE20_levels",
        "is_festival_user",
        "level_use",
        "is_LAST_2YEAR_DD_ACTIVE",
        "cafe_tag_is_mop_available",
        "is_merch_user",
        "p4week_active",
        "is_LAST_1YEAR_DD_ACTIVE",
        "msr_lifestatus",
        "IS_SR_KIT_USER",
        "member_monetary"]

    #  数据预处理
    df_train, df_btest = data_clean2(df)
    df_train = df_train[select_columns]
    df_btest = df_btest[select_columns]

    for cats in catfeatures:
        df_train[cats] = df_train[cats].astype(int)
        df_btest[cats] = df_btest[cats].astype(int)

    # # 抽样
    # df_train = df_train.sample(n=None, frac=0.1, replace=False, weights=None,
    #                            random_state=0, axis=0)
    # df_btest = df_btest.sample(n=None, frac=0.1, replace=False, weights=None,
    #                            random_state=0, axis=0)

    print('正/负', str(len(df_train[df_train[labels] == 1])) + '/' + str(len(df_train[df_train[labels] == 0])))
    t = len(df_train[df_train[labels] == 0]) / len(df_train[df_train[labels] == 1])
    v = len(df_btest[df_btest[labels] == 0]) / len(df_btest[df_btest[labels] == 1])
    print(t, v)

    # #划分训练测试集
    X_train_tra, X_test_tra, df_btest= data_seperate(df_train, df_btest, size=0.3, label=labels, cri=None,
                                                     undeal_column=None)

    # # 划分label
    print(df_train.columns)

    #  划分label
    x_train, y_train = seperate_label(X_train_tra, label=labels)
    x_test, y_test = seperate_label(X_test_tra, label=labels)
    print("x_train", x_train.shape)

    #SBS
    # sbs = SBS(rf, k_features=9,test_size=0.3, random_state=0,scoring=f1_score)
    # sbs.fit(x.drop(["studentNo", "teacherId"], axis=1),y)
    # k_feat = [len(k) for k in sbs.subsets_]
    # import matplotlib.pyplot as plt
    # plt.plot(k_feat, sbs.scores_, marker='o')
    # plt.ylim([0.7, 1.1])
    # plt.ylabel('Accuracy')
    # plt.xlabel('Number of features')
    # plt.grid()
    # plt.tight_layout()
    # # plt.savefig('./sbs.png', dpi=300)
    # plt.show()

    #RFE
    cout = Counter(y_train)
    tt = cout[0] / cout[1] - 23
    sample_weigh = np.where(y_train == 0, 1, tt)
    # clfs = RandomForestClassifier(n_estimators=23, max_depth=6, max_features="auto", random_state=5, n_jobs=-1,
    #                               class_weight={0: 1, 1: tt - 23}
    #                               )
    dt_score = make_scorer(f1_score, pos_label=1)
    # selector = RFECV(clfs, step=1, cv=5, scoring=dt_score, n_jobs=-1)
    # selector = selector.fit(x_train, y_train)
    # print("=============RFECV_RF=============")
    # print("查看哪些特征是被选择的", selector.support_)  #查看哪些特征是被选择的
    # print("被筛选的特征数量", selector.n_features_)
    # print("特征排名", selector.ranking_)

    clfs = XGBClassifier(
        max_depth=7,
        min_child_weight=1,
        learning_rate=0.01,
        n_estimators=20,
        objective='binary:logistic',
        gamma=0,
        max_delta_step=0,
        subsample=1,
        colsample_bytree=1,
        colsample_bylevel=1,
        reg_alpha=0,
        reg_lambda=0,
        scale_pos_weight=tt-2,
        seed=1,
        missing=None,
        use_label_encoder=False,
        random_state=5,
    )
    selector = RFECV(clfs, step=1, cv=5, scoring=dt_score, n_jobs=-1)
    selector = selector.fit(x_train, y_train)
    print("=============RFECV_XGBOOST=============")
    print("查看哪些特征是被选择的", selector.support_)  # 查看哪些特征是被选择的
    print("被筛选的特征数量", selector.n_features_)
    print("特征排名", selector.ranking_)
    columns = x_train.columns
    selects = [columns[i] for i, j in enumerate(selector.support_) if j]
    cat_features = [
        "is_festival_user",
        "is_LAST_2YEAR_DD_ACTIVE",
        "cafe_tag_is_mop_available",
        "IS_SR_KIT_USER",
    ]
    x_train = x_train[selects]
    x_test = x_test[selects]
    x_btest = df_btest[selects]

    adaboost_model(x_train, x_test,
              y_train, y_test,
              x_btest, df_btest[labels])
    lr_model(x_train, x_test,
              y_train, y_test,
              x_btest, df_btest[labels])
    gbdt_mdoel(x_train, x_test,
              y_train, y_test,
              x_btest, df_btest[labels])

    xgb_model(x_train, x_test,
              y_train, y_test,
              x_btest, df_btest[labels])
    lgb_model(x_train, x_test,
              y_train, y_test,
              x_btest, df_btest[labels])
    cat_boost_model(x_train, x_test,
              y_train, y_test,
              x_btest, df_btest[labels],
                    cat_features=cat_features)


    # from sklearn.feature_selection import RFECV
    # x = df_train.copy()
    # clf1 = RandomForestClassifier()
    # clf2 = GradientBoostingClassifier()
    # clf3 = XGBClassifier()
    # dt_score = make_scorer(precision_score, pos_label=1)
    # label = "new_new_isSuccess"
    # unuse_column = ["studentNo", "teacherId"]
    # models = [clf1],
    # scoring = dt_score
    # from collections import Counter
    # y = x[label]
    # cout = Counter(y)
    # t = cout[0] / cout[1]
    # sample_weigh = np.where(y_train == 0, 1, t)
    # unuse_columns = [label] + unuse_column
    # columns = pd.DataFrame(x.drop(unuse_columns, axis=1).columns).rename(columns={0: "features"})
    # i = 1
    # for clf in models:
    #     selector = RFECV(estimator=clf[0], step=1, cv=5, scoring=scoring,n_jobs=-1)
    #     selector = selector.fit(x.drop(unuse_columns, axis=1), y)
    #     neme = "model_" + str(i)
    #     sl = pd.DataFrame(selector.support_).rename(columns={0: neme})
    #     columns = pd.concat([columns, sl], axis=1)
    #     i = i + 1
    #
    # for colm in columns.columns[1:]:
    #     columns[colm] = np.where(columns[colm] == False, 0, 1)
    # # columns = np.where(columns==True,1,0)
    #
    # sum = 0
    # for j in range(len(models)):
    #     sum = sum + columns.iloc[:, j + 1]
    # columns["sum"] = sum
    #
    # columns_select = columns[columns["sum"] > len(models) / 2]
    # select_features = list(columns_select["features"])
    #
    # x_select = pd.concat(
    #     [pd.DataFrame(x[unuse_columns]).reset_index(drop=True), x[select_features].reset_index(drop=True)],
    #     axis=1)
    #
    # df_btest = pd.concat([df_btest[unuse_columns].reset_index(drop=True),
    #                       df_btest[select_features].reset_index(drop=True)], axis=1)