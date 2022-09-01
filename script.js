'use strict';

const canvasHeight = 256;
const canvasWidth = 256;
const classNames = ['Open', 'Closed', 'Healthy', 'Tumor', 'Epilepsy'];
const inputDiv = document.getElementById('inputDiv');
const inputFileName = 'python/release/eeg-classification-example-data.txt';
const signalFileReader = new FileReader();
let csvDataset;
let csvDatasetMax;
let csvDatasetMin;
let line;
let model;

function disableUI(argument) {
	const nodes = document.getElementById('inputControlDiv').getElementsByTagName('*');
	for(let i = 0; i < nodes.length; i++){
		nodes[i].disabled = argument;
	}
}

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

async function loadModel(predictFunction) {
	const loadModelFunction = tf.loadGraphModel;
	model = await loadModelFunction('python/release/resnet34-1D/model.json', {
		onProgress: function (fraction) {
			document.getElementById('modelDownloadFractionDiv').textContent = `Downloading model, please wait ${Math.round(100*fraction)}%.`;
			if (fraction == 1) {
				document.getElementById('modelDownloadFractionDiv').textContent = 'Model downloaded.';
			}
			disableUI(true);
		}
	});
	predictFunction();
	disableUI(false);
}

async function predictView() {
	if (csvDataset === undefined) {
		return;
	}
	if (model === undefined) {
		return;
	}
	let csvDatasetTmp = csvDataset.expandDims(0).expandDims(2);
	csvDatasetTmp = tf.image.resizeBilinear(csvDatasetTmp, [1, model.inputs[0].shape[2]]);
	csvDatasetTmp = csvDatasetTmp.reshape([1, 1, model.inputs[0].shape[2]]);
	const modelOutput = await model.executeAsync(csvDatasetTmp);
	const classProbabilities = modelOutput.softmax().mul(100).arraySync();
	document.getElementById('outputDiv').textContent = '';
	for (let i = 0; i < classProbabilities[0].length; i++) {
		let elementDiv = document.createElement('div');
		elementDiv.textContent = `${classNames[i]}: ${(classProbabilities[0][i]).toFixed(2)}%`
		document.getElementById('outputDiv').append(elementDiv);
	}
}

function signalLoadView() {
	const files = event.currentTarget.files;
	if (files[0]) {
		signalFileReader.readAsText(files[0]);
	}
}

signalFileReader.onload = function() {
	drawSignal(signalFileReader.result);
	predictView();
};

const inputSvg = d3.select('#inputDiv')
	.append('svg')
	.attr('viewBox', [0, 0, canvasWidth, canvasHeight]);
inputSvg.append('path')
	.attr('id', 'pathInput')
	.style('fill', 'none')
	.style('stroke', 'blue');
d3.select('#inputDiv')
	.call(d3.drag()
		.on('start', (event) => {
			event.on('drag', () => {
				const buffer = tf.buffer(csvDataset.shape, csvDataset.dtype, csvDataset.dataSync());
				const x = window.event.clientX - inputDiv.getBoundingClientRect().x
				const y = window.event.clientY - inputDiv.getBoundingClientRect().y
				buffer.set(csvDatasetMax - csvDatasetMax*y/canvasHeight, Math.round(csvDataset.size*x/canvasWidth));
				tf.dispose(csvDataset);
				csvDataset = buffer.toTensor();
				d3.select('#pathInput')
					.attr('d', line(csvDataset.arraySync()));
				predictView();
			});
		}));

fetch(inputFileName)
	.then(response => response.text())
	.then((text) => {
		drawSignal(text);
		predictView();
	})
loadModel(predictView);