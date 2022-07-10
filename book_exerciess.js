const repl = require('repl');

function sigmoid(z) {
    return 1 / (1 + Math.E ** -z);
}


function neuron(w, x, b) {
    return w * x + b;
}


global.sigmoid = sigmoid;
global.neuron = neuron;
repl.start({useColors: true, useGlobal: true});
