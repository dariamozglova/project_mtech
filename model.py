import joblib
import pandas as pd

from main import process_text


def get_results(text:pd.Series) -> pd.Series:
  readed_model = joblib.load('model.joblib')
  proc_text = process_text(text)
  pred = readed_model.predict(proc_text)
  result = pd.Series(pred)
  result.name = 'class_predicted'
  return result

# if __name__ == '__main__':
#     from sklearn.metrics import f1_score
#     X = pd.read_pickle('test_data.xz')
#     y_pred = get_results(X)
#     y_true = pd.read_pickle('test_y.xz')
#     print(f"f1 с новыми фичами, TF-IDFVectoriser: {f1_score(y_true, y_pred, average='weighted')}")