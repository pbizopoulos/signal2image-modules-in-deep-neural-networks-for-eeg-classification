from pyclientsideml import generate_page


def main():
    ml_type = 'signal-classification'
    model_dirs = [
            'https://raw.githubusercontent.com/pbizopoulos/signal2image-modules-in-deep-neural-networks-for-eeg-classification-tfjs/master/alexnet-1D/model.json',
            'https://raw.githubusercontent.com/pbizopoulos/signal2image-modules-in-deep-neural-networks-for-eeg-classification-tfjs/master/lenet-1D/model.json',
            'https://raw.githubusercontent.com/pbizopoulos/signal2image-modules-in-deep-neural-networks-for-eeg-classification-tfjs/master/resnet18-1D/model.json',
            'https://raw.githubusercontent.com/pbizopoulos/signal2image-modules-in-deep-neural-networks-for-eeg-classification-tfjs/master/resnet34-1D/model.json',
            'https://raw.githubusercontent.com/pbizopoulos/signal2image-modules-in-deep-neural-networks-for-eeg-classification-tfjs/master/resnet50-1D/model.json']
    class_names = ['Open', 'Closed', 'Healthy', 'Tumor', 'Epilepsy']
    input_filename = 'https://raw.githubusercontent.com/pbizopoulos/signal2image-modules-in-deep-neural-networks-for-eeg-classification/master/docs/example-signal-1.txt'
    title = 'EEG signal classification demo'
    description = 'NOT FOR MEDICAL USE. Choose a EEG csv file (.txt,.csv) and classify epilepsy using a DNN.'
    url = 'https://github.com/pbizopoulos/signal2image-modules-in-deep-neural-networks-for-eeg-classification'
    block_width = 256
    block_height = 256
    generate_page(ml_type, model_dirs, class_names, input_filename, title, description, url, block_width, block_height)


if __name__ == '__main__':
    main()
