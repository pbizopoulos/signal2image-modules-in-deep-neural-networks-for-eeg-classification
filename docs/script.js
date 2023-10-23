"use strict";
const canvasHeight = 256;
const canvasWidth = 256;
const classNameArray = ["Open", "Closed", "Healthy", "Tumor", "Epilepsy"];
const inputDiv = document.getElementById("input-div");
const inputFileName =
	"https://raw.githubusercontent.com/pbizopoulos/signal2image-modules-in-deep-neural-networks-for-eeg-classification/main/latex/python/prm/eeg-classification-example-data.txt";
const modelDownloadDiv = document.getElementById("model-download-div");
const modelDownloadProgress = document.getElementById(
	"model-download-progress",
);
const outputDiv = document.getElementById("output-div");
const signalFileReader = new FileReader();
const signalInputFile = document.getElementById("signal-input-file");
let csvDataset;
let csvDatasetMax;
let csvDatasetMin;
let line;
let model;
signalFileReader.onload = signalFileReaderOnLoad;
signalInputFile.onchange = signalInputFileOnChange;

function drawSignal(text) {
	const array = text.match(/\d+(?:\.\d+)?/g).map(Number);
	csvDataset = tf.tensor(array);
	csvDatasetMax = csvDataset.max().arraySync();
	csvDatasetMin = csvDataset.min().arraySync();
	const x = d3
		.scaleLinear()
		.domain([0, csvDataset.size])
		.range([0, canvasWidth]);
	const y = d3
		.scaleLinear()
		.domain([csvDatasetMin, csvDatasetMax])
		.range([canvasHeight, 0]);
	line = d3
		.line()
		.x((d, i) => x(i))
		.y((d) => y(d));
	d3.select("#path-input").attr("d", line(csvDataset.arraySync()));
}

async function loadModel(predictFunction) {
	const loadModelFunction = tf.loadGraphModel;
	model = await loadModelFunction(
		"https://raw.githubusercontent.com/pbizopoulos/signal2image-modules-in-deep-neural-networks-for-eeg-classification/main/latex/python/prod/resnet34-1D/model.json",
		{
			onProgress: (fraction) => {
				modelDownloadProgress.value = fraction;
				if (fraction === 1) {
					modelDownloadDiv.style.display = "none";
				}
			},
		},
	);
	predictFunction();
}

async function predictView() {
	if (csvDataset === undefined) {
		return;
	}
	if (model === undefined) {
		return;
	}
	let csvDatasetTmp = csvDataset.expandDims(0).expandDims(2);
	csvDatasetTmp = tf.image.resizeBilinear(csvDatasetTmp, [
		1,
		model.inputs[0].shape[2],
	]);
	csvDatasetTmp = csvDatasetTmp.reshape([1, 1, model.inputs[0].shape[2]]);
	const modelOutput = await model.executeAsync(csvDatasetTmp);
	const classProbabilityArray = modelOutput.softmax().mul(100).arraySync()[0];
	outputDiv.textContent = "";
	for (let i = 0; i < classProbabilityArray.length; i++) {
		const elementDiv = document.createElement("div");
		elementDiv.textContent = `${classNameArray[i]}: ${classProbabilityArray[
			i
		].toFixed(2)}%`;
		outputDiv.append(elementDiv);
	}
}

function signalFileReaderOnLoad() {
	drawSignal(signalFileReader.result);
	predictView();
}

function signalInputFileOnChange() {
	const files = event.currentTarget.files;
	if (files[0]) {
		signalFileReader.readAsText(files[0]);
	}
}

const inputSvg = d3
	.select("#input-div")
	.append("svg")
	.attr("viewBox", [0, 0, canvasWidth, canvasHeight]);
inputSvg
	.append("path")
	.attr("id", "path-input")
	.style("fill", "none")
	.style("stroke", "blue");
d3.select("#input-div").call(
	d3.drag().on("start", (event) => {
		event.on("drag", () => {
			const buffer = tf.buffer(
				csvDataset.shape,
				csvDataset.dtype,
				csvDataset.dataSync(),
			);
			const x = window.event.clientX - inputDiv.getBoundingClientRect().x;
			const y = window.event.clientY - inputDiv.getBoundingClientRect().y;
			buffer.set(
				csvDatasetMax - (csvDatasetMax * y) / canvasHeight,
				Math.round((csvDataset.size * x) / canvasWidth),
			);
			tf.dispose(csvDataset);
			csvDataset = buffer.toTensor();
			d3.select("#path-input").attr("d", line(csvDataset.arraySync()));
			predictView();
		});
	}),
);
fetch(inputFileName)
	.then((response) => response.text())
	.then((text) => {
		drawSignal(text);
		predictView();
	});
loadModel(predictView);
