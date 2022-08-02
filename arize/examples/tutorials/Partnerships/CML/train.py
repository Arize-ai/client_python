import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from arize.api import Client
import datetime

# Set random seed
seed = 42

################################
########## DATA PREP ###########
################################

# Load in the data
df = pd.read_csv(
    "https://storage.googleapis.com/arize-assets/fixtures/wine_quality.csv"
)

# Split into train and test sections
y = df.pop("quality")
X_train, X_test, y_train, y_test = train_test_split(
    df, y, test_size=0.2, random_state=seed
)

#################################
########## MODELLING ############
#################################

# Fit a model on the train section
regr = RandomForestRegressor(max_depth=2, random_state=seed)
regr.fit(X_train, y_train)

# Report training set score
train_score = regr.score(X_train, y_train) * 100
# Report test set score
test_score = regr.score(X_test, y_test) * 100
y_pred = regr.predict(X_test)

# Write scores to a file
with open("metrics.txt", "w") as outfile:
    outfile.write("Training variance explained: %2.1f%%\n" % train_score)
    outfile.write("Test variance explained: %2.1f%%\n" % test_score)

#############################################
########## Arize AI Validation Sample ############
#############################################

SPACE_KEY = "SPACE_KEY"
API_KEY = "API_KEY"
model_name = "validation-wine-model-cicd"


datetime_rightnow = datetime.datetime.today()
model_version_id_now = "train_validate_" + datetime_rightnow.strftime(
    "%m_%d_%Y__%H_%M_%S"
)
id_df = pd.DataFrame([str(id) + model_version_id_now for id in X_test.index])
arize_client = Client(
    space_key=SPACE_KEY, api_key=API_KEY, uri="https://devr.arize.com/v1"
)
tfuture = arize_client.log(
    model_id=model_name,
    model_version=model_version_id_now,
    features=X_test,
    prediction_id=id_df,
    prediction_label=pd.DataFrame(y_pred),
)
tfuture = arize_client.log(
    model_id=model_name,
    model_version=model_version_id_now,
    prediction_id=id_df,
    actual_label=pd.DataFrame(y_test),
)

##########################################
##### PLOT FEATURE IMPORTANCE ############
##########################################
# Calculate feature importance in random forest
importances = regr.feature_importances_
labels = df.columns
feature_df = pd.DataFrame(
    list(zip(labels, importances)), columns=["feature", "importance"]
)
feature_df = feature_df.sort_values(
    by="importance",
    ascending=False,
)

# image formatting
axis_fs = 18  # fontsize
title_fs = 22  # fontsize
sns.set(style="whitegrid")

ax = sns.barplot(x="importance", y="feature", data=feature_df)
ax.set_xlabel("Importance", fontsize=axis_fs)
ax.set_ylabel("Feature", fontsize=axis_fs)  # ylabel
ax.set_title("Random forest\nfeature importance", fontsize=title_fs)

plt.tight_layout()
plt.savefig("feature_importance.png", dpi=120)
plt.close()


##########################################
############ PLOT RESIDUALS  #############
##########################################

y_pred = regr.predict(X_test) + np.random.normal(0, 0.25, len(y_test))
y_jitter = y_test + np.random.normal(0, 0.25, len(y_test))
res_df = pd.DataFrame(list(zip(y_jitter, y_pred)), columns=["true", "pred"])

ax = sns.scatterplot(x="true", y="pred", data=res_df)
ax.set_aspect("equal")
ax.set_xlabel("True wine quality", fontsize=axis_fs)
ax.set_ylabel("Predicted wine quality", fontsize=axis_fs)  # ylabel
ax.set_title("Residuals", fontsize=title_fs)

# Make it pretty- square aspect ratio
ax.plot([1, 10], [1, 10], "black", linewidth=1)
plt.ylim((2.5, 8.5))
plt.xlim((2.5, 8.5))

plt.tight_layout()
plt.savefig("residuals.png", dpi=120)
