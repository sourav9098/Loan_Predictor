import pickle
def load_model():
    with open("models/best_home_loan_model.pkl","rb")as f:
        return pickle.load(f)
def predict_crop(model,features):
    return model.predict(features)[0]    