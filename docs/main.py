from pyjsclient import pyjsclient


def main():
    model_dir = [
            'https://raw.githubusercontent.com/pbizopoulos/signal2image-modules-in-deep-neural-networks-for-eeg-classification-tfjs/master/alexnet-1D/model.json',
            'https://raw.githubusercontent.com/pbizopoulos/signal2image-modules-in-deep-neural-networks-for-eeg-classification-tfjs/master/lenet-1D/model.json',
            'https://raw.githubusercontent.com/pbizopoulos/signal2image-modules-in-deep-neural-networks-for-eeg-classification-tfjs/master/resnet18-1D/model.json',
            'https://raw.githubusercontent.com/pbizopoulos/signal2image-modules-in-deep-neural-networks-for-eeg-classification-tfjs/master/resnet34-1D/model.json',
            'https://raw.githubusercontent.com/pbizopoulos/signal2image-modules-in-deep-neural-networks-for-eeg-classification-tfjs/master/resnet50-1D/model.json']
    problem_type = 'signal classification'
    class_names = ['Open', 'Closed', 'Healthy', 'Tumor', 'Epilepsy']
    title = 'EEG signal classification demo'
    description = 'NOT FOR MEDICAL USE. Choose a EEG csv file (.txt,.csv) and classify epilepsy using a DNN.'
    url = 'https://github.com/pbizopoulos/signal2image-modules-in-deep-neural-networks-for-eeg-classification'
    block_width = 256
    block_height = 256
    pyjsclient(model_dir, problem_type, class_names, title, description, url, block_width, block_height)


if __name__ == '__main__':
    main()
