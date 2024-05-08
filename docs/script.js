const canvasHeight = 256;
const canvasWidth = 256;
const classNameArray = ["Open", "Closed", "Healthy", "Tumor", "Epilepsy"];
const informationButton = document.getElementById("information-button");
const informationDialog = document.getElementById("information-dialog");
const inputDiv = document.getElementById("input-div");
const inputFileName =
	"https://raw.githubusercontent.com/pbizopoulos/signal2image-modules-in-deep-neural-networks-for-eeg-classification/main/docs/prm/eeg-classification-example-data.txt";
const loadingDialog = document.getElementById("loading-dialog");
const outputDiv = document.getElementById("output-div");
const signalFileReader = new FileReader();
const signalInputFile = document.getElementById("signal-input-file");

let csvDataset;
let csvDatasetMax;
let csvDatasetMin;
let line;
let session;
signalFileReader.onload = signalFileReaderOnLoad;
signalInputFile.onchange = signalInputFileOnChange;
informationButton.addEventListener("click", () => {
	informationDialog.showModal();
});

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
	loadingDialog.showModal();
	session = await ort.InferenceSession.create("./prm/model.onnx");
	loadingDialog.close();
	predictFunction();
}

async function predictView() {
	if (csvDataset === undefined) {
		return;
	}
	if (session === undefined) {
		return;
	}
	let csvDatasetTmp = csvDataset.expandDims(0).expandDims(2);
	csvDatasetTmp = tf.image.resizeBilinear(csvDatasetTmp, [1, 178]);
	csvDatasetTmp = csvDatasetTmp.reshape([1, 1, 178]);
	const tensorA = new ort.Tensor(
		"float32",
		csvDatasetTmp.dataSync(),
		[1, 1, 178],
	);
	const feeds = { "input.1": tensorA };
	const results = await session.run(feeds);
	const modelOutput = results["29"].cpuData;
	const classProbabilityArray = tf.softmax(modelOutput).mul(100).arraySync();
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
