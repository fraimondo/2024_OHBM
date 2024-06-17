# %%
from junifer.storage import HDF5FeatureStorage
from julearn.api import run_cross_validation
from julearn.pipeline import PipelineCreator
import pandas as pd
import seaborn as sns
from sklearn.svm import LinearSVC
from pathlib import Path

t_path = Path(__file__).parent

storage = HDF5FeatureStorage(t_path / "data/aomicid1000_brainprint.hdf5")

df_eigen = storage.read_df("FreeSurfer_brainprint_eigenvalues")
df_areas = storage.read_df("FreeSurfer_brainprint_areas")
df_volumes = storage.read_df("FreeSurfer_brainprint_volumes")
df_volumes = df_volumes.reset_index().set_index("subject")
df_volumes["gm_vol"] = df_volumes["lh-pial-2d"] + df_volumes["rh-pial-2d"]

df_demographics = pd.read_csv(t_path / "data/participants.tsv", sep="\t")
df_demographics.rename(columns={"participant_id": "subject"}, inplace=True)
df_demographics = df_demographics.set_index("subject")

columns = [
    "4th-Ventricle", "Brain-Stem", "Left-Cerebellum-White-Matter",
    "Left-Cerebellum-Cortex", "Left-Thalamus-Proper", "Left-Caudate",
    "Left-Putamen", "Left-Pallidum", "Left-Hippocampus", "Left-Amygdala",
    "Left-Accumbens-area", "Left-VentralDC", "Right-Cerebellum-White-Matter",
    "Right-Cerebellum-Cortex", "Right-Thalamus-Proper", "Right-Caudate",
    "Right-Putamen", "Right-Pallidum", "Right-Hippocampus", "Right-Amygdala",
    "Right-Accumbens-area", "Right-VentralDC", "lh-white-2d", "rh-white-2d",
    "lh-pial-2d","rh-pial-2d",
]

df_data = df_eigen[columns].unstack(-1).reset_index().set_index("subject")
df_data.columns = df_data.columns.map(
    lambda x: x if isinstance(x, str) else f"{x[0]}_{x[1]}"
)
df_data = df_data.join(df_demographics)
df_data = df_data.join(df_volumes["gm_vol"])


X = [x for x in df_data.columns if any(x.startswith(y) for y in columns)]
X_types = {"continuous": [".*"]}

creator = PipelineCreator(problem_type="classification")
creator.add("zscore")
creator.add(
    "svm",
    C=(0.001, 100, "log-uniform"),
)

search_params = {
    "kind": "optuna",
    "cv": 5,
}
scores, model, inspector = run_cross_validation(
    X=X,
    y="sex",
    data=df_data,
    X_types=X_types,
    search_params=search_params,
    model=creator,
    return_train_score=True,
    return_inspector=True,
    cv=5,
)

predictions = inspector.folds.predict()
to_merge = df_data[["sex", "gm_vol"]].iloc[predictions.index]
predictions.index = to_merge.index
to_plot = pd.concat([predictions, to_merge], axis=1)

to_plot["correct"] = to_plot["repeat0_p0"] == to_plot["target"]
sns.boxplot(data=to_plot, x="sex", hue="correct", y="gm_vol")

# %%
import matplotlib.pyplot as plt
to_plot["correct"] = to_plot["repeat0_p0"] == to_plot["target"]
sns.boxplot(data=to_plot, x="sex", hue="correct", y="gm_vol")
plt.savefig("bias.pdf")
# %%
