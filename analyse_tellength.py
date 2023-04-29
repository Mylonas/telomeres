import pandas as pd  # Tables library
import matplotlib.pyplot as plt  # Plotting library
import seaborn as sns  # Plotting library
import glob  # Path names library
import numpy as np  # Maths library
import matplotlib.pyplot as plt
from nltk import DecisionTreeClassifier
from scipy.stats import linregress
from sklearn import neighbors
from scipy import stats  # Stats library
import sklearn  # Machine learning and AI library
from sklearn import linear_model
from sklearn.ensemble._hist_gradient_boosting.gradient_boosting import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV, RANSACRegressor, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics
from collections import defaultdict
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestRegressor, ExtraTreesClassifier, VotingRegressor
# A function to define the root mean squared error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Imputer
from sklearn.ensemble import IsolationForest
# from sklearn.ensemble import StackingRegressor
from mlxtend.regressor import StackingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.linear_model import RidgeCV
import time
from sklearn.model_selection import cross_validate, cross_val_predict
# The function ``plot_regression_results`` is used to plot the predicted and
# true targets.
from sklearn.tree import DecisionTreeRegressor
import warnings
import hvplot.pandas

warnings.filterwarnings("ignore")
from sklearn.svm import SVR, LinearSVR


def plot_regression_results(ax, y_true, y_pred, title, scores, elapsed_time):
    """Scatter plot of the predicted vs true targets."""
    ax.plot([y_true.min(), y_true.max()],
            [y_true.min(), y_true.max()],
            '--r', linewidth=2)
    ax.scatter(y_true, y_pred, alpha=0.2)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    ax.set_xlim([y_true.min(), y_true.max()])
    ax.set_ylim([y_true.min(), y_true.max()])
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                          edgecolor='none', linewidth=0)
    ax.legend([extra], [scores], loc='upper left')
    title = title + '\n Evaluation in {:.2f} seconds'.format(elapsed_time)
    ax.set_title(title)


def rmse(predictions, targets):  # root mean squared error
    return np.sqrt(np.mean((predictions - targets) ** 2))


# A function to calculate leaveOneOut cross validation
def leaveOneOut(X, y, model):
    loo = LeaveOneOut()

    X = np.array(X)
    y = np.array(y)

    predicted = []
    actual = []

    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # rf = RandomForestRegressor(n_estimators=100)
        rf = model
        rf.fit(X_train, y_train)

        predicted.append(rf.predict(X_test)[0])  # returns 1D array
        actual.append(y_test[0])

    predicted = np.array(predicted)
    actual = np.array(actual)

    # Print separating ine
    print("." * 80)

    print("Predictions leaveOneOut -----")
    print({"RMSE": rmse(predicted, actual)})
    print({"Rho": stats.pearsonr(predicted, actual)[0]})
    # print({"R^2": stats.linregress(predicted, actual)[2] ** 2}, "\n")
    print({"R^2": metrics.r2_score(actual, predicted)}, "\n")

    return predicted

# A function to fit a model for k-fold cross validation
def k_fold(X, y, model):
    # lm = linear_model.LinearRegression()
    lm = model
    model = lm.fit(X, y)
    predictions = cross_val_predict(model, X, y, cv=20)

    # r2 = metrics.r2_score(y, predictions)

    # Print separating line
    print("." * 80)

    print("Predictions K-Fold -----")
    print("RMSE on test data: ", rmse(predictions, y))
    print("Pearsons rho: ", stats.pearsonr(predictions, y))
    # slope, intercept, r_value, p_value, std_err = stats.linregress(predictions, y)
    # print("R^2: ", r_value ** 2, "\n")
    print({"R^2": metrics.r2_score(y, predictions)}, "\n")
    # print("K-Fold cross-validation - Predicted Accuracy:", r2, "\n")


# https://www.analyticsvidhya.com/blog/2018/08/k-nearest-neighbor-introduction-regression-python/
# K-Nearest Neighbors Regression
def nearest_neighbour_regression(X, y):
    # Create train and test set
    X_train, X_test = train_test_split(X, test_size=0.2)
    y_train, y_test = train_test_split(y, test_size=0.2)

    rmse_val = []  # to store rmse values for different k
    for K in range(30):  # error rate
        K = K + 1
        model = neighbors.KNeighborsRegressor(n_neighbors=K)

        model.fit(X_train, y_train)  # fit the model
        pred = model.predict(X_test)  # make prediction on test set
        error = np.sqrt(mean_squared_error(y_test, pred))  # calculate rmse
        rmse_val.append(error)  # store rmse values
    # print('RMSE value for k= ', K, 'is:', error)  # error rate

    # plotting the rmse values against k values
    curve = pd.DataFrame(rmse_val)  # elbow curve
    curve.plot()

    params = {'n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]}

    knn = neighbors.KNeighborsRegressor()

    model = GridSearchCV(knn, params, cv=10)
    model.fit(X_train, y_train)
    print(model.best_params_)


# https://www.geeksforgeeks.org/python-decision-tree-regression-using-sklearn/
# A function to create a flowchart tree structure of the model
def decision_tree_regression(X, y):
    # create a regressor object
    regressor = DecisionTreeRegressor(random_state=0)

    # fit the regressor with X and Y data
    regressor.fit(X, y)

    # arange for creating a range of values
    # from min value of X to max value of X
    # with a difference of 0.01 between two
    # consecutive values
    X_grid = np.arange(-10, 100, float(0.01))

    # reshape for reshaping the data into
    # a len(X_grid)*1 array, i.e. to make
    # a column out of the X_grid values
    X_grid = X_grid.reshape((len(X_grid), 1))

    # Generate plot scatter plot for original data
    plt.scatter(X, y, color='red')

    # plot predicted data
    plt.plot(X_grid, regressor.predict(X_grid), color='blue')

    # specify title
    plt.title('Telomre Length (Decision Tree Regression)')

    # specify X axis label
    plt.xlabel('Actual telomerecat')
    plt.xlim(0, 8)

    # specify Y axis label
    plt.ylabel('Predictions')
    plt.ylim(0, 8)

    # show the plot
    plt.show()


# https://towardsdatascience.com/feature-selection-techniques-in-machine-learning-with-python-f24e7da3f36e
# A method to dind the parameter/s that contribute the most to our model
def find_greatest_impact(df2, model):
    # model = ExtraTreesClassifier()
    X = df2[["Telseq", "Telomerecat", "chromothripsis", "tumor"]]  # True Values
    y = df2[["tumor"]]  # True Values

    model.fit(X, y)
    print(model.feature_importances_)  # use inbuilt class feature_importances of tree based classifiers
    # plot graph of feature importances for better visualization
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat_importances.nlargest(10).plot(kind='barh')
    plt.show()


# https://scikit-learn.org/stable/auto_examples/ensemble/plot_stack_predictors.html#sphx-glr-auto-examples-ensemble-plot-stack-predictors-py
# combine 3 learners (linear and non-linear) and use a ridge regressor to combine their outputs together.
def stack_regressor(X, y):
    estimators = [
        ('Random Forest', RandomForestRegressor(random_state=42)),
        ('Lasso', LassoCV()),
        ('Gradient Boosting', HistGradientBoostingRegressor(random_state=0))
    ]
    stacking_regressor = StackingRegressor(
        estimators=estimators, final_estimator=RidgeCV()
    )

    fig, axs = plt.subplots(2, 2, figsize=(9, 7))
    axs = np.ravel(axs)

    for ax, (name, est) in zip(axs, estimators + [('Stacking Regressor', stacking_regressor)]):
        start_time = time.time()
        score = cross_validate(est, X, y,
                               scoring=['r2', 'neg_mean_absolute_error'],
                               n_jobs=-1, verbose=0)
        elapsed_time = time.time() - start_time

        y_pred = cross_val_predict(est, X, y, n_jobs=-1, verbose=0)
        plot_regression_results(
            ax, y, y_pred,
            name,
            (r'$R^2={:.2f} \pm {:.2f}$' + '\n' + r'$MAE={:.2f} \pm {:.2f}$')
                .format(np.mean(score['test_r2']),
                        np.std(score['test_r2']),
                        -np.mean(score['test_neg_mean_absolute_error']),
                        np.std(score['test_neg_mean_absolute_error'])),
            elapsed_time)

    plt.suptitle('Single predictors versus stacked predictors')
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()


# Outlier detection
def outlier_detection(X, y):
    # Outlier detection
    clf = IsolationForest(n_estimators=10, warm_start=True)
    clf.fit(X)  # fit 10 trees
    clf.set_params(n_estimators=10)  # add 20 more trees
    preds = clf.fit_predict(X)  # fit the added trees
    # print("Predictions here")
    # print(preds)

    # Print separating line
    print("." * 80)
    print("Number of samples analysed: ", len(preds))
    global count
    count = 0
    for i in range(len(preds)):
        if preds[i] == -1:
            count = count + 1
    print("Samples dropped: ", count)

    for i in range(len(preds)):
        if preds[i] == -1:
            print(X.iloc[[i]])
            X.drop(X.index[i], inplace=True)
            y.drop(y.index[i], inplace=True)


# https://scikit-learn.org/stable/auto_examples/svm/plot_svm_regression.html
# SVM regression
def svm(df2, X, y):
    # X = df2[["Telseq", "Telomerecat", "tumor"]]
    X = df2[["tumor"]]
    X = np.array(X)
    y = np.array(y)

    # Fit regression model
    svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
    svr_lin = SVR(kernel='linear', C=100, gamma='auto')
    svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1,
                   coef0=1)

    # Look at the results
    lw = 2

    svrs = [svr_rbf, svr_lin, svr_poly]
    kernel_label = ['RBF', 'Linear', 'Polynomial']
    model_color = ['m', 'c', 'g']

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 10), sharey=True)
    for ix, svr in enumerate(svrs):
        axes[ix].plot(X, svr.fit(X, y).predict(X), color=model_color[ix], lw=lw,
                      label='{} model'.format(kernel_label[ix]))
        axes[ix].scatter(X[svr.support_], y[svr.support_], facecolor="none",
                         edgecolor=model_color[ix], s=50,
                         label='{} support vectors'.format(kernel_label[ix]))
        axes[ix].scatter(X[np.setdiff1d(np.arange(len(X)), svr.support_)],
                         y[np.setdiff1d(np.arange(len(X)), svr.support_)],
                         facecolor="none", edgecolor="k", s=50,
                         label='other training data')
        axes[ix].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
                        ncol=1, fancybox=True, shadow=True)

    fig.text(0.5, 0.04, 'data', ha='center', va='center')
    fig.text(0.06, 0.5, 'target', ha='center', va='center', rotation='vertical')
    fig.suptitle("Support Vector Regression", fontsize=14)
    plt.show()


# This function is for generating new plots
def scatter_plot(X, y):
    stats = linregress(X, y)

    m = stats.slope
    b = stats.intercept

    # Change the default figure size
    plt.figure(figsize=(10, 10))

    # Change the default marker for the scatter from circles to x's
    plt.scatter(X, y, marker='x')

    # Set the linewidth on the regression line to 3px
    plt.plot(X, m * X + b, "red", linewidth=3)

    # Add x and y lables, and set their font size
    plt.xlabel("TelomereCat (Kb)", fontsize=20)
    plt.ylabel("STELA (kb)", fontsize=20)

    # Set the font size of the number lables on the axes
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    # plt.savefig("python-linear-reg-custom.png")
    plt.show()


# Remove all samples that are more than 5kb
def cap_at_five(df2, df3):
    # Removing all values with a kb more than 5 (removes 10 samples)
    df2['Telseq'].mask(df2['Telseq'].between(5, 100), inplace=True)
    df2['Telomerecat'].mask(df2['Telomerecat'].between(5, 100), inplace=True)

    # Imputing NaN values with the median
    df2['Telseq'] = df2['Telseq'].fillna((df2['Telseq'].median()))
    df2['Telomerecat'] = df2['Telomerecat'].fillna((df2['Telomerecat'].median()))

    # Removing all values with a kb more than 5 (removes 10 samples)
    df3['TelSeq_Normal'].mask(df3['TelSeq_Normal'].between(5, 100), inplace=True)
    df3['TelSeq_Tumor'].mask(df3['TelSeq_Tumor'].between(5, 100), inplace=True)

    df3['TelomereCat_Normal'].mask(df3['TelomereCat_Normal'].between(5, 100), inplace=True)
    df3['TelomereCat_Tumor'].mask(df3['TelomereCat_Tumor'].between(5, 100), inplace=True)

    # Imputing NaN values with the median
    df3['TelSeq_Normal'] = df3['TelSeq_Normal'].fillna((df3['TelSeq_Normal'].median()))
    df3['TelSeq_Tumor'] = df3['TelSeq_Tumor'].fillna((df3['TelSeq_Tumor'].median()))

    df3['TelomereCat_Normal'] = df3['TelomereCat_Normal'].fillna((df3['TelomereCat_Normal'].median()))
    df3['TelomereCat_Tumor'] = df3['TelomereCat_Tumor'].fillna((df3['TelomereCat_Tumor'].median()))


# Add all tumour and healthy tissue data together in one line
def combine_datasets(df2):
    # initialize list of lists
    data = [[]]
    # Create the pandas DataFrame
    df10 = pd.DataFrame(data)

    # print(len(df2))

    # This fixes error with indexing values and missing values
    df2.to_csv('Step1.csv', index=False)
    df2 = pd.read_csv('Step1.csv')

    # Loop to create new dataset
    for i in range(0, len(df2), 2):
        if i == len(df2) - 1:
            break
        else:
            id_1 = df2.at[i, 'sample']
            id_2 = df2.at[i + 1, 'sample']

            TelSeq_Normal = df2.at[i, 'Telseq']
            TelSeq_Tumor = df2.at[i + 1, 'Telseq']

            TelomereCat_Normal = df2.at[i, 'Telomerecat']
            TelomereCat_Tumor = df2.at[i + 1, 'Telomerecat']

            y = df2.at[i + 1, 'y']

            i = i / 2
            df10.loc[i, 'sample'] = id_1 + '/' + id_2
            df10.loc[i, 'TelSeq_Normal'] = TelSeq_Normal
            df10.loc[i, 'TelSeq_Tumor'] = TelSeq_Tumor
            df10.loc[i, 'TelomereCat_Normal'] = TelomereCat_Normal
            df10.loc[i, 'TelomereCat_Tumor'] = TelomereCat_Tumor
            df10.loc[i, 'y'] = y

            # df2 = pd.read_csv('Step1.csv')
    df10.to_csv('df_tumor.csv', index=False)
    return df10


# Get the data sorted
def plot_stela_tel():
    # Convert to DB.. style names
    df = pd.read_csv("./sample_pairs.csv")  # load the dataset as a pandas data frame
    df_tumour = pd.read_csv("./Telomere_lengths_from_tools.csv")  # load the dataset as a pandas data frame
    df_tumour.columns = df_tumour.columns.to_series().apply(lambda x: x.strip())
    # print(df)

    # Array to load chromothripsis data
    chromothripsis_list = ['DB163', 'DB179', 'DB221', 'DB211', 'DB225', 'DB167', 'DB171', 'DB227', 'DB181', 'DB154',
                           'DB156', 'DB213', 'DB185', 'DB159', 'DB148', 'DB217', 'DB223', 'DB144', 'DB169', 'DB199',
                           'DB197', 'DB191', 'DB195', 'DB177', 'DB229', 'DB203', 'DB219']

    # Get stela results
    name_stela_tn = {}
    for item in ("normal", "tumor"):
        name_stela_tn.update({k: (v, item) for k, v in zip(df[item + "_db"], df[item + "_stela"])})

    res = []
    # Get tel seq results
    for item in glob.glob("raw_data/telseq/*.telseq"):
        df = pd.read_csv(item, sep="\t")  # load the dataset as a pandas data frame
        sample = item.split("\\")[-1].split(".")[0]
        stela_length, tumor_or_normal = name_stela_tn[sample]

        if tumor_or_normal == "normal":
            tumor = 0

        elif tumor_or_normal == "tumor":
            tumor = 1

        chromothripsis = 0
        for i in chromothripsis_list:
            if i == sample:
                chromothripsis = 1

        res.append({"STELA (kb)": stela_length,
                    "Predicted tel (kb)": df["LENGTH_ESTIMATE"].iloc[0],
                    "TEL0": df["TEL0"].iloc[0] / 100000000,
                    "Only Telomeric (kb)": df["TEL0"].iloc[0] / 100000000,
                    "Model": "Telseq", "kind": tumor_or_normal,
                    "tumor": tumor,
                    "chromothripsis": chromothripsis,
                    "sample": sample})

    # Get telomerecat results
    # load the data-set as a pandas data frame
    for idx, r in pd.read_csv("raw_data/telomerecat/telomerecat_length_1542385578.csv", index_col=None).iterrows():
        sample = r["Sample"].split(".")[0]
        stela_length, tumor_or_normal = name_stela_tn[sample]

        if tumor_or_normal == "normal":
            tumor = 0
        elif tumor_or_normal == "tumor":
            tumor = 1

        chromothripsis = 0
        for i in chromothripsis_list:
            if i == sample:
                chromothripsis = 1

        res.append({"STELA (kb)": stela_length,
                    "Predicted tel (kb)": r["Length"] / 1e3,
                    "F1": r["F1"] / 1e3,
                    "Only Telomeric (kb)": r["F1"] / 1e3,
                    "Model": "Telomerecat", "kind": tumor_or_normal,
                    "tumor": tumor,
                    "chromothripsis": chromothripsis,
                    "sample": sample})

    # Note dropna, samples with no tel length removed
    df = pd.DataFrame.from_records(res).sort_values("sample")  # .dropna()

    # df.to_csv('df.csv', index=False)
    # Overall prediction for root mean square error
    print("Overall ------")
    print("RMSE on test data: ", rmse(df["Predicted tel (kb)"], df["STELA (kb)"]))
    #    print("Pearsons rho: ", stats.pearsonr(df["Predicted tel (kb)"], df["STELA (kb)"]))
    slope, intercept, r_value, p_value, std_err = stats.linregress(df["Predicted tel (kb)"], df["STELA (kb)"])
    print("R^2: ", r_value ** 2)
    print()

    # Some stats for different software and groups
    for (model, kind), dm in df.groupby(["Model", "kind"]):
        print(model, kind, "------")

        print("RMSE on test data: ", rmse(dm["Predicted tel (kb)"], dm["STELA (kb)"]))
        #        print("Pearsons rho: ", stats.pearsonr(dm["Predicted tel (kb)"], dm["STELA (kb)"]))
        slope, intercept, r_value, p_value, std_err = stats.linregress(dm["Predicted tel (kb)"], dm["STELA (kb)"])
        print("R^2: ", r_value ** 2)
        print()

    # Generate plot
    # g = sns.lmplot("STELA (kb)", "Predicted tel (kb)", data=df, hue="Model", col="kind", legend_out=False, ci=68,
    # truncate=True, robust=True, markers=["o", "x"], height=4, aspect=1, row="chromothripsis")
    # g.set(ylim=(0, 8))
    # g.set(xlim=(0, 8))
    # plt.savefig("./telseq_telcat.pdf")
    # plt.show()

    # g = sns.lmplot(["CompuTel", "TelSeq"], "STELA (kb)", data=df_tumour, legend_out=False, ci=68,
    #                truncate=True, robust=True, height=4, aspect=1)

    # g.set(ylim=(0, 8))
    # g.set(xlim=(0, 8))
    df_tumour = df_tumour.reset_index()

    # hvplot.show(df_tumour.hvplot())

    return df


# Apply regression and cross-validation techniques
def convert_to_features_format(df):
    # This step is to convert the table into a 'wide' format, rather than long format
    # print("DF columns", df.columns)

    d = defaultdict(dict)
    for idx, r in df.iterrows():
        d[r["sample"]].update({r["Model"]: r["Predicted tel (kb)"], "y": r["STELA (kb)"], "sample": r["sample"],
                               "kind": r["kind"], "tumor": r["tumor"], "chromothripsis": r["chromothripsis"]})
        # print(r["sample"])

        for item in ("TEL0", "F1"):
            if r[item] == r[item]:
                d[r["sample"]].update({item: r[item]})
                # print(item, r[item])
            continue

    df2 = pd.DataFrame.from_records(list(d.values()))
    df2 = df2.dropna()

    # X = df2[["tumor", "TEL0", "F1"]]   # True Values
    X = df2[["Telseq", "Telomerecat", "tumor"]]  # True Values
    y = df2["y"]  # Predictions

    # Add all tumour and healthy tissue data together in one line
    df3 = combine_datasets(df2)
    #X = df3[["TelSeq_Normal", "TelSeq_Tumor", "TelomereCat_Normal", "TelomereCat_Tumor"]]  # True Values
    #y = df3["y"]  # Predictions

    # Remove all samples that are more than 5kb
    # cap_at_five(df2, df3)

    # Outlier detection and removal
    # outlier_detection(X, y)

    ###############################################################
    # Different regression models

    # lm = linear_model.LinearRegression()
    # lm = linear_model.Ridge(alpha=.5)
    lm = neighbors.KNeighborsRegressor()
    # lm = DecisionTreeRegressor(random_state=0)
    # lm = RANSACRegressor(random_state=0) # negative scores
    # lm = VotingRegressor([('lr', LinearRegression()), ('rf', RandomForestRegressor(n_estimators=10, random_state=1))])

    # not working, error with the import statement
    # from sklearn.ensemble import StackingRegressor
    # lm = StackingRegressor([('lr', RidgeCV()), ('svr', LinearSVR(random_state=42))], RandomForestRegressor(n_estimators=10,random_state = 42))

    # SVM regression
    # svm(df2, X, y)

    ###############################################################

    ###############################################################
    # Different cross validation models

    # K-Nearest Neighbors Regression
    # nearest_neighbour_regression(X, y)

    # Plotting greatest impact parameter
    # find_greatest_impact(df3, ExtraTreesClassifier())

    # Combine the strengths of the different regressor
    # stack_regressor(X,y) # not working

    # A method to calculate leaveOneOut cross-validation
    leaveOneOut(X, y, lm)

    # A method for k-fold cross validation
    k_fold(X, y, lm)

    ###############################################################

    # A function to create a flowchart tree structure of the model
    # decision_tree_regression(df2[["Telseq"]], y)
    # decision_tree_regression(df2[["Telseq"]], y)

    # Fit a model
    # lm = linear_model.LinearRegression()
    model = lm.fit(X, y)
    pred = lm.predict(X)
    print(pred)

    print("Predictions -----")
    print("RMSE on test data: ", rmse(pred, y))
    print("Pearsons rho: ", stats.pearsonr(pred, y))
    # slope, intercept, r_value, p_value, std_err = stats.linregress(pred, y)
    # print("R^2: ", r_value ** 2)
    r_square = metrics.r2_score(y, pred)
    print({"R^2": r_square}, "\n")

    lr = linear_model.LinearRegression()
    predicted = cross_val_predict(lr, X, y, cv=20)

    fig, ax = plt.subplots()
    ax.scatter(y, predicted)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    ax.set_xlabel('STELA (kb)')
    ax.set_ylabel('Predicted (kb)')
    #plt.show()

    # hvplot.show(df3.hvplot(kind = 'scatter'))
    df3 = plot_tel_and_tumor()
    X = df3[['Predicted tel (Kb)', 'chromothripsis']]
    y = df3[['STELA (kb)']]

    predicted = cross_val_predict(lm, X, y, cv=20)
    fig, ax = plt.subplots()
    ax.scatter(y, predicted, edgecolors=(0, 0, 0))
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    ax.set_xlabel('STELA (kb)')
    ax.set_ylabel('Predicted (kb)')
    plt.show()

    # Print plots of tumor only data
    # scatter_plot(X, y)

    return r_square


# Get the data sorted
def plot_tumor_only():
    # Convert to DB.. style names
    df = pd.read_csv("./df_tumor.csv")  # load the dataset as a pandas data frame

    res = []
    # Get stela results
    name_stela_tn = {}
    for item in range(len(df)):  # ("Normal", "Tumor"):
        # name_stela_tn.update({k: (v, item) for k, v in zip(df["TelSeq_" + item], df["TelomereCat_" + item])})
        sample = df.at[item, 'sample']

        stela_length = df.at[item, 'y']

        TelSeq_Normal = df.at[item, 'TelSeq_Normal']
        TelSeq_Tumor = df.at[item, 'TelSeq_Tumor']

        TelomereCat_Normal = df.at[item, 'TelomereCat_Normal']
        TelomereCat_Tumor = df.at[item, 'TelomereCat_Tumor']

        res.append({"STELA (kb)": stela_length,
                    "Predicted tel (Kb)": TelSeq_Normal,
                    "Model": "TelSeq",
                    "kind": "normal",
                    "tumor": 0,
                    "sample": sample})

        res.append({"STELA (kb)": stela_length,
                    "Predicted tel (Kb)": TelSeq_Tumor,
                    "Model": "TelSeq",
                    "kind": "tumor",
                    "tumor": 1,
                    "sample": sample})

        res.append({"STELA (kb)": stela_length,
                    "Predicted tel (Kb)": TelomereCat_Normal,
                    "Model": "TelomereCat",
                    "kind": "normal",
                    "tumor": 0,
                    "sample": sample})

        res.append({"STELA (kb)": stela_length,
                    "Predicted tel (Kb)": TelomereCat_Tumor,
                    "Model": "TelomereCat",
                    "kind": "tumor",
                    "tumor": 1,
                    "sample": sample})

    # Note dropna, samples with no tel length removed
    df = pd.DataFrame.from_records(res).sort_values("sample")  # .dropna()

    # Overall prediction for root mean square error
    # print("Overall ------")
    # print("RMSE on test data: ", rmse(df["Predicted tel (kb)"], df["STELA (kb)"]))
    #    print("Pearsons rho: ", stats.pearsonr(df["Predicted tel (kb)"], df["STELA (kb)"]))
    # slope, intercept, r_value, p_value, std_err = stats.linregress(df["Predicted tel (kb)"], df["STELA (kb)"])
    # print("R^2: ", r_value ** 2)
    # print()

    # Some stats for different software and groups
    # for (model, kind), dm in df.groupby(["Model", "kind"]):
    #    print(model, kind, "------")

    #    print("RMSE on test data: ", rmse(dm["Predicted tel (kb)"], dm["STELA (kb)"]))
    #        print("Pearsons rho: ", stats.pearsonr(dm["Predicted tel (kb)"], dm["STELA (kb)"]))
    #    slope, intercept, r_value, p_value, std_err = stats.linregress(dm["Predicted tel (kb)"], dm["STELA (kb)"])
    #    print("R^2: ", r_value ** 2)
    #    print()

    # Generate plot
    g = sns.lmplot("STELA (kb)", "Predicted tel (Kb)", data=df, hue="Model", col="kind", legend_out=False, ci=68,
                   truncate=True, robust=True, markers=["o", "x"], height=4, aspect=1)
    g.set(ylim=(0, 8))
    g.set(xlim=(0, 8))
    # plt.savefig("./telseq_telcat.pdf")
    plt.show()

    # g = sns.lmplot(["CompuTel", "TelSeq"], "STELA (kb)", data=df_tumour, legend_out=False, ci=68,
    #                truncate=True, robust=True, height=4, aspect=1)

    # g.set(ylim=(0, 8))
    # g.set(xlim=(0, 8))
    # df_tumour = df_tumour.reset_index()

    # hvplot.show(df_tumour.hvplot())

    return df


def plot_tel_and_tumor():
    df = pd.read_csv("./Step1.csv")  # load the dataset as a pandas data frame

    res = []
    # Get stela results
    name_stela_tn = {}
    for item in range(len(df)):  # ("Normal", "Tumor"):
        # name_stela_tn.update({k: (v, item) for k, v in zip(df["TelSeq_" + item], df["TelomereCat_" + item])})
        sample = df.at[item, 'sample']

        stela_length = df.at[item, 'y']
        tumor = df.at[item,'tumor']
        chromothripsis = df.at[item, 'chromothripsis']
        if tumor==0:
            TelSeq_Normal = df.at[item, 'Telseq']
            TelomereCat_Normal = df.at[item, 'Telomerecat']

            res.append({"STELA (kb)": stela_length,
                        "Predicted tel (Kb)": TelSeq_Normal,
                        "Model": "TelSeq",
                        "kind": "normal",
                        "chromothripsis":chromothripsis,
                        "sample": sample})

            res.append({"STELA (kb)": stela_length,
                        "Predicted tel (Kb)": TelomereCat_Normal,
                        "Model": "TelomereCat",
                        "kind": "normal",
                        "chromothripsis": chromothripsis,
                        "sample": sample})

        if tumor == 1:
            TelSeq_Tumor = df.at[item, 'Telseq']
            TelomereCat_Tumor = df.at[item, 'Telomerecat']

            res.append({"STELA (kb)": stela_length,
                        "Predicted tel (Kb)": TelSeq_Tumor,
                        "Model": "TelSeq",
                        "kind": "tumor",
                        "chromothripsis": chromothripsis,
                        "sample": sample})

            res.append({"STELA (kb)": stela_length,
                        "Predicted tel (Kb)": TelomereCat_Tumor,
                        "Model": "TelomereCat",
                        "kind": "tumor",
                        "chromothripsis": chromothripsis,
                        "sample": sample})

    # Note dropna, samples with no tel length removed
    df = pd.DataFrame.from_records(res).sort_values("sample")  # .dropna()
    df['Predicted tel (Kb)'].mask(df['Predicted tel (Kb)'].between(5, 100), inplace=True)
    df['Predicted tel (Kb)'] = df['Predicted tel (Kb)'].fillna((df['Predicted tel (Kb)'].median()))

    # Generate plot
    #g = sns.lmplot("STELA (kb)", "Predicted tel (Kb)", data=df, hue="Model", col="chromothripsis", legend_out=False, ci=68,
#                   truncate=True, robust=True, markers=["o", "x"], height=4, aspect=1, row="kind")
    #g.set(ylim=(0, 8))
    #g.set(xlim=(0, 8))
    # plt.savefig("./telseq_telcat.pdf")
    #plt.show()

    # g = sns.lmplot(["CompuTel", "TelSeq"], "STELA (kb)", data=df_tumour, legend_out=False, ci=68,
    #                truncate=True, robust=True, height=4, aspect=1)

    # g.set(ylim=(0, 8))
    # g.set(xlim=(0, 8))
    # df_tumour = df_tumour.reset_index()

    # hvplot.show(df_tumour.hvplot())

    return df

# Main script calling the base functions of the pipeline
def main():
    df = plot_stela_tel()
    r_square = convert_to_features_format(df)
    #plot_tumor_only()
    #plot_tel_and_tumor()
    count = 0
    # while r_square < 0.68 and count < 10:
    #   r_square = convert_to_features_format(df)
    #  count = count + 1


main()
