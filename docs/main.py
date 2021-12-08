from pyclientsideml import generate_page_signal_classification


def main():
    model_dirs = [
            'https://raw.githubusercontent.com/pbizopoulos/signal2image-modules-in-deep-neural-networks-for-eeg-classification-tfjs/master/alexnet-1D/model.json',
            'https://raw.githubusercontent.com/pbizopoulos/signal2image-modules-in-deep-neural-networks-for-eeg-classification-tfjs/master/lenet-1D/model.json',
            'https://raw.githubusercontent.com/pbizopoulos/signal2image-modules-in-deep-neural-networks-for-eeg-classification-tfjs/master/resnet18-1D/model.json',
            'https://raw.githubusercontent.com/pbizopoulos/signal2image-modules-in-deep-neural-networks-for-eeg-classification-tfjs/master/resnet34-1D/model.json',
            'https://raw.githubusercontent.com/pbizopoulos/signal2image-modules-in-deep-neural-networks-for-eeg-classification-tfjs/master/resnet50-1D/model.json']
    model_names = [
            'alexnet-1D',
            'lenet-1D',
            'resnet18-1D',
            'resnet34-1D',
            'resnet50-1D']
    class_names = ['Open', 'Closed', 'Healthy', 'Tumor', 'Epilepsy']
    title = 'EEG signal classification demo'
    description = 'NOT FOR MEDICAL USE. Choose a EEG csv file (.txt,.csv) and classify epilepsy using a DNN.'
    url = 'https://github.com/pbizopoulos/signal2image-modules-in-deep-neural-networks-for-eeg-classification'
    block_width = 256
    block_height = 256
    input_filename = 'https://raw.githubusercontent.com/pbizopoulos/signal2image-modules-in-deep-neural-networks-for-eeg-classification/master/docs/example-signal.txt'
    generate_page_signal_classification(model_dirs, model_names, class_names, title, description, url, block_width, block_height, input_filename)


if __name__ == '__main__':
    main()
