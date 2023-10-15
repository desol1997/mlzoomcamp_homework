import pickle


def read_file(file_name, mode='rb'):
    with open(file_name, mode=mode) as f:
        result = pickle.load(f)
    
    return result


def predict_prob(client, model, dv):
    X = dv.transform([client])
    y = model.predict_proba(X)[0, 1]

    return y
