from utils import read_file, predict_prob


if __name__ == '__main__':
    model_file_name = 'model1.bin'
    dv_file_name = 'dv.bin'

    model = read_file(model_file_name)
    dv = read_file(dv_file_name)

    client = {"job": "retired", "duration": 445, "poutcome": "success"}

    print(predict_prob(client, model=model, dv=dv))
