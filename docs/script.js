const canvasWidth = 256;
const canvasHeight = 256;
const inputFilename = 'https://raw.githubusercontent.com/pbizopoulos/signal2image-modules-in-deep-neural-networks-for-eeg-classification/master/docs/eeg-classification-example-data.txt';

function drawSignal(text) {
	const array = text.match(/\d+(?:\.\d+)?/g).map(Number);
	csvDataset = tf.tensor(array);
	csvDatasetMax = csvDataset.max().arraySync();
	csvDatasetMin = csvDataset.min().arraySync();
	const x = d3.scaleLinear()
		.domain([0, csvDataset.size])
		.range([0, canvasWidth]);
	const y = d3.scaleLinear()
		.domain([csvDatasetMin, csvDatasetMax])
		.range([canvasHeight, 0]);
	line = d3.line()
		.x((d,i) => x(i))
		.y(d => y(d));
	d3.select('#pathInput')
		.attr('d', line(csvDataset.arraySync()));
}

const classNames = ['Open', 'Closed', 'Healthy', 'Tumor', 'Epilepsy'];
if (inputFilename) {
	fetch(inputFilename)
		.then(response => response.text())
		.then((text) => {
			drawSignal(text);
			predictView();
		})
}

let csvDataset;
let csvDatasetMax;
let csvDatasetMin;
const svgInput = d3.select('#divInput')
	.append('svg')
	.attr('viewBox', [0, 0, canvasWidth, canvasHeight]);
svgInput.append('path')
	.attr('id', 'pathInput')
	.style('fill', 'none')
	.style('stroke', 'blue');
d3.select('#divInput')
	.call(d3.drag()
		.on('start', (event) => {
			event.on('drag', () => {
				const buffer = tf.buffer(csvDataset.shape, csvDataset.dtype, csvDataset.dataSync());
				buffer.set(csvDatasetMin + csvDatasetMax*(canvasHeight - window.event.pageY)/canvasHeight, Math.round(csvDataset.size*(window.event.pageX - canvasWidth)/canvasWidth));
				tf.dispose(csvDataset);
				csvDataset = buffer.toTensor();
				d3.select('#pathInput')
					.attr('d', line(csvDataset.arraySync()));
				predictView();
			});
		}));

let line;
const signalFileReader = new FileReader();
signalFileReader.onload = () => {
	drawSignal(signalFileReader.result);
	predictView();
};

function signalLoadView() {
	const files = event.currentTarget.files;
	if (files[0]) {
		signalFileReader.readAsText(files[0]);
	}
}

async function predictView() {
	if (csvDataset === undefined) {
		return;
	}
	if (model === undefined) {
		return;
	}
	csvDatasetTmp = csvDataset.expandDims(0).expandDims(2);
	csvDatasetTmp = tf.image.resizeBilinear(csvDatasetTmp, [1, model.inputs[0].shape[2]]);
	csvDatasetTmp = csvDatasetTmp.reshape([1, 1, model.inputs[0].shape[2]]);
	const modelOutput = await model.executeAsync(csvDatasetTmp);
	const classProbabilities = modelOutput.softmax().mul(100).arraySync();
	document.getElementById('divOutput').textContent = '';
	for (let i = 0; i < classProbabilities[0].length; i++) {
		let divElement = document.createElement('div');
		divElement.textContent = `${classNames[i]}: ${(classProbabilities[0][i]).toFixed(2)}%`
		document.getElementById('divOutput').append(divElement);
	}
}

function disableUI(argument) {
	const nodes = document.getElementById('divInputControl').getElementsByTagName('*');
	for(let i = 0; i < nodes.length; i++){
		nodes[i].disabled = argument;
	}
}

let model;
async function loadModel(predictFunction) {
	const loadModelFunction = tf.loadGraphModel;
	model = await loadModelFunction('https://raw.githubusercontent.com/pbizopoulos/signal2image-modules-in-deep-neural-networks-for-eeg-classification/master/docs/resnet34-1D/model.json', {
		onProgress: function (fraction) {
			document.getElementById('divModelDownloadFraction').textContent = `Downloading model, please wait ${Math.round(100*fraction)}%.`;
			if (fraction == 1) {
				document.getElementById('divModelDownloadFraction').textContent = 'Model downloaded.';
			}
			disableUI(true);
		}
	});
	predictFunction();
	disableUI(false);
}
loadModel(predictView);
