/*-----
这是对 Andrej Karpathy "原子级" Python 实现的直接移植。旨在仅使用 Node.js 标准库 + ES5语法，演示 Transformer 的核心算法——自动求导、注意力机制和优化器。

运行方式：
  node microgpt.js            # 正常训练 + 推理
  node microgpt.js --trace    # 训练 + 推理，并生成 viz.html 可视化文件
------*/

var fs = require('fs');
var path = require('path');
var child_process = require('child_process');

// --trace 模式：训练结束后生成 viz.html
var doTrace = process.argv.indexOf('--trace') !== -1;

var _seed = 42;
function rand() {
    _seed = (_seed * 16807) % 2147483647;
    return (_seed - 1) / 2147483646;
}

function randn() {
    var u = rand();
    var v = rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function r4(x) { return Math.round(x * 10000) / 10000; }   // 保留4位小数

//数据加载（对应 Python os + urllib）
var inputPath = path.join(__dirname, 'input.txt');
if (!fs.existsSync(inputPath)) {
    console.log('input.txt not found, downloading from GitHub...');
    var url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt';
    child_process.execSync('curl -L ' + url + ' -o ' + inputPath, { stdio: 'inherit' });
    console.log('Download finished.');
}

var text = fs.readFileSync(inputPath, 'utf8');
var lines = text.split('\n');

var docs = [];
for (var i = 0; i < lines.length; i++) {
    var s = lines[i].trim();
    if (s.length > 0) {
        docs.push(s);
    }
}

//shuffle
for (var i = docs.length - 1; i > 0; i--) {
    var j = Math.floor(rand() * (i + 1));
    var tmp = docs[i];
    docs[i] = docs[j];
    docs[j] = tmp;
}

console.log('num docs:', docs.length);


//Tokenizer
var chars = {};
for (var i = 0; i < docs.length; i++) {
    var d = docs[i];
    for (var k = 0; k < d.length; k++) {
        chars[d[k]] = true;
    }
}

var uchars = [];
for (var c in chars) {
    uchars.push(c);
}
uchars.sort();

var BOS = uchars.length;
var vocabSize = uchars.length + 1;

// [改进1] charToId：O(n) → O(1)
var charToIdMap = {};
for (var i = 0; i < uchars.length; i++) {
    charToIdMap[uchars[i]] = i;
}
function charToId(ch) {
    return charToIdMap.hasOwnProperty(ch) ? charToIdMap[ch] : -1;
}

function idToChar(id) {
    return uchars[id];
}

console.log('vocab size:', vocabSize);


// [改进2] Value 节点唯一 ID
var _nextId = 0;

function Value(data, children, grads) {
    this.id = _nextId++;
    this.data = data;
    this.grad = 0;
    this.children = children || [];
    this.grads = grads || [];
}

function add(a, b) {
    return new Value(a.data + b.data, [a, b], [1, 1]);
}

function mul(a, b) {
    return new Value(a.data * b.data, [a, b], [b.data, a.data]);
}

function neg(a) {
    return mul(a, new Value(-1));
}

function sub(a, b) {
    return add(a, neg(b));
}

function pow(a, p) {
    return new Value(Math.pow(a.data, p), [a], [p * Math.pow(a.data, p - 1)]);
}

function logv(a) {
    return new Value(Math.log(a.data), [a], [1 / a.data]);
}

function expv(a) {
    var e = Math.exp(a.data);
    return new Value(e, [a], [e]);
}

function relu(a) {
    return new Value(a.data > 0 ? a.data : 0, [a], [a.data > 0 ? 1 : 0]);
}


// [改进3] backward：O(n²) → O(1)
function backward(v) {
    var topo = [];
    var visited = {};

    function visit(x) {
        if (visited[x.id]) return;
        visited[x.id] = true;
        for (var i = 0; i < x.children.length; i++) {
            visit(x.children[i]);
        }
        topo.push(x);
    }

    visit(v);
    v.grad = 1;

    for (var i = topo.length - 1; i >= 0; i--) {
        var node = topo[i];
        for (var j = 0; j < node.children.length; j++) {
            node.children[j].grad += node.grads[j] * node.grad;
        }
    }
}

//模型超参数
var nLayer = 1;
var nEmb = 16;
var blockSize = 16;
var nHead = 4;
var headDim = nEmb / nHead;


// [改进4] 参数按层命名，支持 nLayer > 1
function matrix(rows, cols) {
    var m = [];
    for (var i = 0; i < rows; i++) {
        var r = [];
        for (var j = 0; j < cols; j++) {
            r.push(new Value(randn() * 0.08));
        }
        m.push(r);
    }
    return m;
}

var state = {
    wte: matrix(vocabSize, nEmb),
    wpe: matrix(blockSize, nEmb),
    lm_head: matrix(vocabSize, nEmb)
};
for (var li = 0; li < nLayer; li++) {
    state['layer' + li + '.attn_wq'] = matrix(nEmb, nEmb);
    state['layer' + li + '.attn_wk'] = matrix(nEmb, nEmb);
    state['layer' + li + '.attn_wv'] = matrix(nEmb, nEmb);
    state['layer' + li + '.attn_wo'] = matrix(nEmb, nEmb);
    state['layer' + li + '.mlp_fc1'] = matrix(4 * nEmb, nEmb);
    state['layer' + li + '.mlp_fc2'] = matrix(nEmb, 4 * nEmb);
}


function linear(x, w) {
    var out = [];
    for (var i = 0; i < w.length; i++) {
        var sum = new Value(0);
        for (var j = 0; j < x.length; j++) {
            sum = add(sum, mul(w[i][j], x[j]));
        }
        out.push(sum);
    }
    return out;
}

function softmax(xs) {
    var max = xs[0].data;
    for (var i = 1; i < xs.length; i++) {
        if (xs[i].data > max) max = xs[i].data;
    }
    var exps = [];
    var sum = new Value(0);
    for (var i = 0; i < xs.length; i++) {
        var e = expv(sub(xs[i], new Value(max)));
        exps.push(e);
        sum = add(sum, e);
    }
    var probs = [];
    for (var i = 0; i < exps.length; i++) {
        probs.push(mul(exps[i], pow(sum, -1)));
    }
    return probs;
}

function rmsnorm(x) {
    var sum = new Value(0);
    for (var i = 0; i < x.length; i++) {
        sum = add(sum, mul(x[i], x[i]));
    }
    var mean = mul(sum, new Value(1 / x.length));
    var scale = pow(add(mean, new Value(1e-5)), -0.5);
    var out = [];
    for (var i = 0; i < x.length; i++) {
        out.push(mul(x[i], scale));
    }
    return out;
}


// [改进5] gpt 支持 nLayer 层
function gpt(tokenId, posId, keys, values) {
    var tok = state.wte[tokenId];
    var pos = state.wpe[posId];

    var x = [];
    for (var i = 0; i < nEmb; i++) {
        x.push(add(tok[i], pos[i]));
    }
    x = rmsnorm(x);

    for (var li = 0; li < nLayer; li++) {
        var x_res = x;
        x = rmsnorm(x);

        var q = linear(x, state['layer' + li + '.attn_wq']);
        var k = linear(x, state['layer' + li + '.attn_wk']);
        var v = linear(x, state['layer' + li + '.attn_wv']);

        keys[li].push(k);
        values[li].push(v);

        var attn_out = [];
        for (var j = 0; j < nEmb; j++) {
            attn_out.push(new Value(0));
        }

        for (var h = 0; h < nHead; h++) {
            var hs = h * headDim;
            var attn_logits = [];

            for (var t = 0; t < keys[li].length; t++) {
                var score = new Value(0);
                for (var j2 = 0; j2 < headDim; j2++) {
                    score = add(score, mul(q[hs + j2], keys[li][t][hs + j2]));
                }
                score = mul(score, new Value(1 / Math.sqrt(headDim)));
                attn_logits.push(score);
            }

            var weights = softmax(attn_logits);

            for (var j = 0; j < headDim; j++) {
                var wsum = new Value(0);
                for (var t = 0; t < values[li].length; t++) {
                    wsum = add(wsum, mul(weights[t], values[li][t][hs + j]));
                }
                attn_out[hs + j] = wsum;
            }
        }

        var proj = linear(attn_out, state['layer' + li + '.attn_wo']);
        x = [];
        for (var i = 0; i < nEmb; i++) {
            x.push(add(proj[i], x_res[i]));
        }

        x_res = x;
        x = rmsnorm(x);
        x = linear(x, state['layer' + li + '.mlp_fc1']);
        for (var i = 0; i < x.length; i++) {
            x[i] = relu(x[i]);
        }
        x = linear(x, state['layer' + li + '.mlp_fc2']);
        for (var i = 0; i < nEmb; i++) {
            x[i] = add(x[i], x_res[i]);
        }
    }

    return linear(x, state.lm_head);
}


//Adam 优化器
var lr = 0.01;
var beta1 = 0.85;
var beta2 = 0.99;
var eps = 1e-8;

var params = [];
for (var k in state) {
    var mat = state[k];
    for (var i = 0; i < mat.length; i++) {
        for (var j = 0; j < mat[i].length; j++) {
            params.push(mat[i][j]);
        }
    }
}

console.log('num params:', params.length);

var m = [];
var vv = [];
for (var i = 0; i < params.length; i++) {
    m.push(0);
    vv.push(0);
}


// ——— Trace 数据结构 ———
// lossHistory: 每步都记录（step, word, loss, lr）
// paramHistory: 每5步记录一次参数的 grad 和 value
var trackedMatrices = doTrace ? [
    { id: 'wte_0',    label: 'wte["' + uchars[0] + '"]',  matrix: 'wte',               row: 0 },
    { id: 'wpe_0',    label: 'wpe[pos=0]',                 matrix: 'wpe',               row: 0 },
    { id: 'wq_0',     label: 'layer0.W_Q[0]',              matrix: 'layer0.attn_wq',    row: 0 },
    { id: 'lmh_0',    label: 'lm_head["' + uchars[0] + '"]', matrix: 'lm_head',         row: 0 }
] : null;

var lossHistory = doTrace ? [] : null;
var paramHistory = doTrace ? [] : null;

function getTrackedRow(matrixName, rowIdx) {
    return state[matrixName][rowIdx];
}


//训练循环
var steps = 1000;

for (var step = 0; step < steps; step++) {
    var doc = docs[step % docs.length];
    var tokens = [BOS];
    for (var i = 0; i < doc.length; i++) {
        tokens.push(charToId(doc[i]));
    }
    tokens.push(BOS);

    // [改进6] KV 缓存按层初始化
    var keys = [];
    var values = [];
    for (var li = 0; li < nLayer; li++) {
        keys.push([]);
        values.push([]);
    }

    var n = Math.min(tokens.length - 1, blockSize);
    var losses = [];
    for (var i = 0; i < n; i++) {
        var logits = gpt(tokens[i], i, keys, values);
        var probs = softmax(logits);
        losses.push(neg(logv(probs[tokens[i + 1]])));
    }

    // [改进7] Loss 归一化
    var loss = new Value(0);
    for (var i = 0; i < losses.length; i++) {
        loss = add(loss, losses[i]);
    }
    loss = mul(loss, new Value(1 / n));

    backward(loss);

    // ——— Trace：backward 之后、Adam 之前记录梯度 ———
    var gradSnapshot = null;
    if (doTrace) {
        gradSnapshot = {};
        for (var ti = 0; ti < trackedMatrices.length; ti++) {
            var spec = trackedMatrices[ti];
            var row = getTrackedRow(spec.matrix, spec.row);
            gradSnapshot[spec.id] = row.map(function(v) { return r4(v.grad); });
        }
    }

    var lr_t = lr * (1 - step / steps);
    // [改进8] beta 幂次提到外层
    var beta1_t = 1 - Math.pow(beta1, step + 1);
    var beta2_t = 1 - Math.pow(beta2, step + 1);
    for (var i = 0; i < params.length; i++) {
        m[i] = beta1 * m[i] + (1 - beta1) * params[i].grad;
        vv[i] = beta2 * vv[i] + (1 - beta2) * params[i].grad * params[i].grad;
        var mh = m[i] / beta1_t;
        var vh = vv[i] / beta2_t;
        params[i].data -= lr_t * mh / (Math.sqrt(vh) + eps);
        params[i].grad = 0;
    }

    // ——— Trace：Adam 之后记录参数值 ———
    if (doTrace) {
        lossHistory.push({ s: step + 1, w: doc, l: r4(loss.data), lr: r4(lr_t) });

        if (step % 5 === 0 || step === steps - 1) {
            var paramRecord = { s: step + 1, p: {} };
            for (var ti = 0; ti < trackedMatrices.length; ti++) {
                var spec = trackedMatrices[ti];
                var row = getTrackedRow(spec.matrix, spec.row);
                paramRecord.p[spec.id] = {
                    g: gradSnapshot[spec.id],
                    v: row.map(function(v) { return r4(v.data); })
                };
            }
            paramHistory.push(paramRecord);
        }
    }

    if (step % 20 === 0) {
        process.stdout.write('step ' + (step + 1) + ' / ' + steps + ' | loss ' + loss.data.toFixed(4) + '\r');
    }
}
console.log('');


//推理生成（[改进9] 纯数字 softmax，跳过 autograd）
function sampleLogits(logits, temperature) {
    var max = logits[0].data;
    for (var i = 1; i < logits.length; i++) {
        if (logits[i].data > max) max = logits[i].data;
    }
    var exps = [];
    var sum = 0;
    for (var i = 0; i < logits.length; i++) {
        var e = Math.exp((logits[i].data - max) / temperature);
        exps.push(e);
        sum += e;
    }
    var r = rand();
    var acc = 0;
    for (var i = 0; i < exps.length; i++) {
        acc += exps[i] / sum;
        if (r < acc) return i;
    }
    return BOS;
}

console.log('--- inference ---');
var temperature = 0.5;  // [改进10] temperature 提到循环外

var inferenceResults = [];

for (var s = 0; s < 10; s++) {
    var keys = [];
    var values = [];
    for (var li = 0; li < nLayer; li++) {
        keys.push([]);
        values.push([]);
    }
    var token = BOS;
    var out = [];

    for (var i = 0; i < blockSize; i++) {
        var logits = gpt(token, i, keys, values);
        var next = sampleLogits(logits, temperature);
        if (next === BOS) break;
        out.push(idToChar(next));
        token = next;
    }

    var name = out.join('');
    console.log(name);
    inferenceResults.push(name);
}


// ============================================================
// viz.html 模板生成函数
// ============================================================
function buildVizHtml(trace, snap) {
    var traceJson = JSON.stringify(trace);
    var snapJson = JSON.stringify(snap);

    return '<!DOCTYPE html>\n<html lang="zh-CN">\n<head>\n<meta charset="UTF-8">\n<meta name="viewport" content="width=device-width,initial-scale=1">\n<title>MicroGPT 可视化教学</title>\n<link rel="preconnect" href="https://fonts.googleapis.com">\n<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700;900&display=swap" rel="stylesheet">\n<style>\n' + VIZ_CSS + '\n</style>\n</head>\n<body>\n' + VIZ_HTML + '\n<script>\nconst TRACE=' + traceJson + ';\nconst SNAP=' + snapJson + ';\n' + VIZ_JS + '\n</script>\n</body>\n</html>';
}


var VIZ_CSS = [
':root{--cream:#FFFDF5;--black:#000;--red:#FF6B6B;--yellow:#FFD93D;--violet:#C4B5FD;--white:#fff}',
'*{box-sizing:border-box;margin:0;padding:0}',
'body{font-family:"Space Grotesk",monospace,sans-serif;background:var(--cream);color:var(--black);min-height:100vh}',
'/* ===== HEADER ===== */',
'.hdr{background:var(--black);color:var(--white);padding:16px 24px;display:flex;align-items:center;gap:16px;border-bottom:4px solid var(--black)}',
'.hdr-badge{background:var(--red);border:3px solid var(--white);padding:4px 12px;font-weight:900;font-size:13px;letter-spacing:.1em;box-shadow:4px 4px 0 var(--white)}',
'.hdr-title{font-size:22px;font-weight:900;letter-spacing:-.02em}',
'.hdr-sub{font-size:12px;opacity:.6;margin-left:auto;font-weight:700;letter-spacing:.05em}',
'/* ===== TABS ===== */',
'.tab-nav{display:flex;border-bottom:4px solid var(--black);background:var(--yellow)}',
'.tab-btn{flex:1;padding:14px 8px;font-weight:900;font-size:13px;letter-spacing:.1em;border:none;border-right:4px solid var(--black);background:transparent;cursor:pointer;transition:background 80ms linear}',
'.tab-btn:last-child{border-right:none}',
'.tab-btn:hover{background:var(--red);color:var(--white)}',
'.tab-btn.active{background:var(--black);color:var(--white)}',
'/* ===== PANELS ===== */',
'.tab-panel{display:none;padding:24px;max-width:1400px;margin:0 auto}',
'.tab-panel.active{display:block}',
'/* ===== NEO CARD ===== */',
'.neo-card{background:var(--white);border:4px solid var(--black);box-shadow:6px 6px 0 var(--black);padding:16px;position:relative}',
'.card-label{font-size:11px;font-weight:900;letter-spacing:.15em;text-transform:uppercase;opacity:.5;margin-bottom:6px}',
'.card-val{font-size:28px;font-weight:900;letter-spacing:-.02em;line-height:1}',
'.card-val.sm{font-size:16px}',
'/* ===== TRAINING PANEL ===== */',
'.train-grid{display:grid;grid-template-columns:160px 1fr 320px;gap:20px;align-items:start}',
'@media(max-width:900px){.train-grid{grid-template-columns:1fr;}}',
'.info-col{display:flex;flex-direction:column;gap:12px}',
'.info-col .neo-card{padding:12px 16px}',
'.info-col .card-val{font-size:22px}',
'/* loss chart */',
'.chart-wrap{border:4px solid var(--black);background:var(--white);box-shadow:6px 6px 0 var(--black);padding:8px;margin-bottom:16px}',
'.chart-label{font-size:11px;font-weight:900;letter-spacing:.15em;padding:0 4px 4px;opacity:.5}',
'#loss-svg{width:100%;height:120px;display:block}',
'.loss-line{fill:none;stroke:var(--red);stroke-width:2.5}',
'.loss-dot{fill:var(--black);r:4}',
'/* controls */',
'.ctrl-row{display:flex;gap:10px;margin-bottom:12px;align-items:center}',
'.neo-btn{border:3px solid var(--black);background:var(--yellow);box-shadow:4px 4px 0 var(--black);padding:8px 16px;font-weight:900;font-size:12px;letter-spacing:.1em;cursor:pointer;transition:transform 80ms,box-shadow 80ms}',
'.neo-btn:hover{transform:translate(-2px,-2px);box-shadow:6px 6px 0 var(--black)}',
'.neo-btn:active{transform:translate(2px,2px);box-shadow:2px 2px 0 var(--black)}',
'.neo-btn.red{background:var(--red);color:var(--white)}',
'.neo-btn.dark{background:var(--black);color:var(--white)}',
'.slider-wrap{display:flex;align-items:center;gap:8px;margin-bottom:8px}',
'.slider-wrap label{font-size:11px;font-weight:900;letter-spacing:.1em;white-space:nowrap}',
'input[type=range]{flex:1;accent-color:var(--black);cursor:pointer}',
'.step-readout{font-size:11px;font-weight:700;white-space:nowrap;min-width:70px}',
'/* param tracker */',
'.param-col{display:flex;flex-direction:column;gap:12px}',
'.param-tabs{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:10px}',
'.ptab{border:3px solid var(--black);background:var(--cream);padding:5px 10px;font-size:11px;font-weight:900;letter-spacing:.05em;cursor:pointer;box-shadow:3px 3px 0 var(--black)}',
'.ptab.active{background:var(--black);color:var(--white)}',
'.vecs-row{display:grid;grid-template-columns:1fr 1fr;gap:10px}',
'.vec-panel{border:3px solid var(--black);background:var(--white);padding:10px;box-shadow:4px 4px 0 var(--black)}',
'.vec-title{font-size:10px;font-weight:900;letter-spacing:.15em;margin-bottom:8px;text-transform:uppercase}',
'.bar-item{display:flex;align-items:center;gap:4px;margin-bottom:3px;font-size:10px;font-weight:700}',
'.bar-idx{width:12px;text-align:right;opacity:.4;flex-shrink:0}',
'.bar-track{flex:1;height:12px;background:#eee;border:1px solid #ccc;position:relative;overflow:hidden}',
'.bar-fill{height:100%;position:absolute;top:0;transition:width .15s,left .15s}',
'.bar-fill.pos{background:var(--red);left:50%}',
'.bar-fill.neg{background:#4488ff;right:50%}',
'.bar-num{width:56px;text-align:right;flex-shrink:0;font-size:9px}',
'/* ===== ATTENTION PANEL ===== */',
'.attn-grid{display:grid;grid-template-columns:1fr 280px;gap:20px}',
'@media(max-width:800px){.attn-grid{grid-template-columns:1fr}}',
'.attn-input-row{display:flex;gap:10px;margin-bottom:16px;align-items:center}',
'.attn-input{border:4px solid var(--black);padding:10px 14px;font-size:16px;font-weight:700;font-family:inherit;background:var(--white);width:220px;box-shadow:4px 4px 0 var(--black)}',
'.attn-input:focus{outline:none;background:var(--yellow)}',
'.head-tabs{display:flex;gap:8px;margin-bottom:14px}',
'.htab{border:3px solid var(--black);background:var(--cream);padding:5px 12px;font-size:11px;font-weight:900;cursor:pointer;box-shadow:3px 3px 0 var(--black)}',
'.htab.active{background:var(--violet)}',
'.tok-display{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:16px}',
'.tok-chip{border:3px solid var(--black);padding:5px 10px;font-weight:900;font-size:13px;background:var(--white);box-shadow:3px 3px 0 var(--black)}',
'.tok-chip.bos{background:var(--yellow)}',
'.tok-chip.active-query{background:var(--red);color:var(--white)}',
'.heatmap-outer{overflow-x:auto}',
'.heatmap-table{border-collapse:collapse}',
'.heatmap-table th,.heatmap-table td{border:2px solid var(--black);padding:0;width:36px;height:36px;text-align:center;font-size:10px;font-weight:700;cursor:pointer}',
'.heatmap-table th{background:var(--yellow);font-size:9px;letter-spacing:.05em}',
'.hmcell{display:flex;align-items:center;justify-content:center;width:100%;height:100%;font-size:9px;font-weight:700;transition:all .1s}',
'.heatmap-legend{display:flex;align-items:center;gap:8px;margin-top:8px;font-size:10px;font-weight:700}',
'.legend-grad{width:120px;height:12px;border:2px solid var(--black);background:linear-gradient(to right,var(--cream),var(--red))}',
'.qkv-panel{display:flex;flex-direction:column;gap:10px}',
'.qkv-block{border:3px solid var(--black);background:var(--white);padding:10px;box-shadow:4px 4px 0 var(--black)}',
'.qkv-title{font-size:10px;font-weight:900;letter-spacing:.1em;margin-bottom:6px}',
'.qkv-row{display:flex;gap:2px;flex-wrap:wrap}',
'.qkv-cell{width:20px;height:20px;border:1px solid var(--black);font-size:7px;font-weight:700;display:flex;align-items:center;justify-content:center}',
'/* ===== INFERENCE PANEL ===== */',
'.infer-top{display:flex;gap:16px;align-items:flex-end;margin-bottom:20px;flex-wrap:wrap}',
'.temp-card{min-width:200px}',
'.temp-val{font-size:22px;font-weight:900;margin-top:4px}',
'.prob-wrap{border:4px solid var(--black);background:var(--white);box-shadow:6px 6px 0 var(--black);padding:16px;margin-bottom:16px}',
'.prob-title{font-size:11px;font-weight:900;letter-spacing:.15em;margin-bottom:10px}',
'.prob-bar-item{display:flex;align-items:center;gap:8px;margin-bottom:4px;font-size:11px;font-weight:700}',
'.prob-char{width:20px;text-align:center;font-weight:900;font-size:13px}',
'.prob-track{flex:1;height:18px;background:#eee;border:2px solid var(--black);position:relative}',
'.prob-fill{height:100%;background:var(--red);transition:width .2s;position:absolute;top:0;left:0}',
'.prob-num{width:50px;text-align:right;font-size:10px}',
'.samples-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:10px;margin-bottom:20px}',
'.sample-chip{border:4px solid var(--black);background:var(--yellow);padding:10px 14px;font-weight:900;font-size:15px;box-shadow:5px 5px 0 var(--black);text-transform:uppercase;letter-spacing:.05em}',
'.sample-chip.new{background:var(--red);color:var(--white);animation:pop .3s ease-out}',
'@keyframes pop{0%{transform:scale(.8)}60%{transform:scale(1.1)}100%{transform:scale(1)}}',
'.seq-display{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:12px}',
'.seq-tok{border:3px solid var(--black);padding:4px 8px;font-weight:900;font-size:13px;box-shadow:2px 2px 0 var(--black)}',
'.seq-tok.just-sampled{background:var(--red);color:var(--white);animation:pop .25s ease-out}',
'/* ===== MISC ===== */',
'.section-title{font-size:16px;font-weight:900;letter-spacing:-.01em;margin-bottom:10px;border-bottom:4px solid var(--black);padding-bottom:8px}',
'.hint{font-size:11px;font-weight:700;opacity:.5;margin-top:6px}',
'/* ===== 说明气泡 ===== */',
'.tip{background:var(--yellow);border:3px solid var(--black);padding:8px 14px;font-size:12px;line-height:1.7;margin-bottom:14px;box-shadow:3px 3px 0 var(--black);font-weight:600;border-radius:0}',
'.tip strong{font-weight:900;font-size:13px}',
'.tip-violet{background:var(--violet)}',
'.tip-white{background:var(--white)}',
'/* ===== 训练词元芯片 ===== */',
'.train-sample-wrap{border:4px solid var(--black);background:var(--white);padding:12px 14px;box-shadow:6px 6px 0 var(--black);margin-bottom:14px}',
'.train-sample-lbl{font-size:10px;font-weight:900;letter-spacing:.15em;opacity:.5;margin-bottom:8px}',
'.train-toks{display:flex;flex-wrap:wrap;gap:5px;align-items:center}',
'.ttok{border:3px solid var(--black);padding:6px 10px;font-weight:900;font-size:15px;background:var(--white);box-shadow:3px 3px 0 var(--black);line-height:1}',
'.ttok.t-bos{background:var(--yellow);font-size:11px;letter-spacing:.05em}',
'.ttok.t-ctx{background:var(--cream)}',
'.ttok.t-pred{background:var(--red);color:var(--white);animation:pulse-tok .85s ease-in-out infinite}',
'.ttok.t-sep{border:none;box-shadow:none;font-size:18px;opacity:.3;padding:0 2px;background:transparent;font-weight:900}',
'@keyframes pulse-tok{0%,100%{transform:translateY(0);box-shadow:3px 3px 0 var(--black)}50%{transform:translateY(-2px);box-shadow:5px 6px 0 var(--black)}}',
'/* ===== 梯度下降流程管道 ===== */',
'.pipe-row{display:flex;align-items:center;margin:4px 0 18px;gap:0}',
'.pipe-stage{border:4px solid var(--black);padding:8px 10px;font-size:10px;font-weight:900;text-align:center;letter-spacing:.05em;box-shadow:4px 4px 0 var(--black);background:var(--white);min-width:72px;position:relative;transition:background .35s,color .35s}',
'.pipe-stage .ps-val{display:block;font-size:17px;font-weight:900;letter-spacing:-.02em;margin-top:3px}',
'.ps-hot{background:var(--red)!important;color:var(--white)!important}',
'.ps-warm{background:#f97316!important;color:var(--white)!important}',
'.ps-ok{background:#fbbf24!important;color:var(--black)!important}',
'.ps-cool{background:#22c55e!important;color:var(--white)!important}',
'.pipe-conn{flex:1;height:8px;background:#e5e7eb;border-top:3px solid var(--black);border-bottom:3px solid var(--black);position:relative;overflow:hidden;min-width:16px}',
'.pipe-fill{position:absolute;top:0;left:0;height:100%;width:0;background:var(--red)}',
'.pipe-fill.go{animation:pipeGo .38s ease-out forwards}',
'.pipe-dot{position:absolute;top:50%;transform:translateY(-50%);width:11px;height:11px;border-radius:50%;background:var(--black);left:0;opacity:0}',
'.pipe-dot.go{animation:dotGo .38s ease-in-out forwards}',
'@keyframes pipeGo{from{width:0}to{width:100%}}',
'@keyframes dotGo{0%{left:0;opacity:1}100%{left:calc(100% - 11px);opacity:1}}',
'.loss-chg{font-size:10px;font-weight:900;padding:2px 5px;border:2px solid var(--black);margin-top:4px;display:block}',
'.loss-chg.lc-up{background:var(--red);color:var(--white)}',
'.loss-chg.lc-dn{background:#22c55e;color:var(--white)}',
'/* ===== 注意力分步导览 ===== */',
'.attn-steps{display:flex;gap:7px;margin-bottom:14px;flex-wrap:wrap}',
'.astep{border:3px solid var(--black);padding:7px 14px;font-size:11px;font-weight:900;cursor:pointer;box-shadow:3px 3px 0 var(--black);background:var(--cream);display:flex;align-items:center;gap:6px;transition:transform 80ms,box-shadow 80ms}',
'.astep:hover:not(.active){background:var(--yellow);transform:translate(-1px,-1px);box-shadow:4px 4px 0 var(--black)}',
'.astep.active{background:var(--black);color:var(--white)}',
'.astep-num{width:20px;height:20px;border-radius:50%;background:var(--yellow);color:var(--black);font-size:10px;font-weight:900;display:flex;align-items:center;justify-content:center;flex-shrink:0}',
'.astep.active .astep-num{background:var(--white)}',
'.attn-step-box{border:4px solid var(--black);background:var(--white);padding:14px;box-shadow:6px 6px 0 var(--black);margin-bottom:16px;min-height:90px}',
'.attn-step-title{font-size:10px;font-weight:900;letter-spacing:.12em;margin-bottom:10px;opacity:.5}',
'.score-grid{display:inline-block;border:3px solid var(--black)}',
'.score-row{display:flex}',
'.sc{width:44px;height:30px;border:1px solid rgba(0,0,0,.15);display:flex;align-items:center;justify-content:center;font-size:9px;font-weight:700;transition:background .2s}',
'.sc.sc-hd{background:var(--yellow);border:2px solid rgba(0,0,0,.3);font-size:9px;font-weight:900}',
'/* ===== 推理词链 ===== */',
'.token-chain{display:flex;flex-wrap:wrap;gap:5px;align-items:center;min-height:54px;padding:10px 12px;border:4px solid var(--black);background:var(--white);box-shadow:6px 6px 0 var(--black);margin-bottom:14px}',
'.ctok{border:3px solid var(--black);padding:7px 12px;font-weight:900;font-size:17px;box-shadow:3px 3px 0 var(--black)}',
'.ctok.ct-bos{background:var(--yellow);font-size:11px;letter-spacing:.05em}',
'.ctok.ct-new{background:var(--red);color:var(--white);animation:popIn .3s cubic-bezier(.34,1.56,.64,1)}',
'.ctok.ct-old{background:var(--white)}',
'.ctok.ct-sep{border:none;box-shadow:none;font-size:20px;opacity:.25;background:transparent;padding:0 2px}',
'@keyframes popIn{from{transform:scale(.5);opacity:0}to{transform:scale(1);opacity:1}}',
'/* ===== 推理控制行 ===== */',
'.infer-ctrl{display:flex;gap:12px;align-items:flex-end;flex-wrap:wrap;margin-bottom:16px}',
'.temp-card2{border:4px solid var(--black);padding:10px 14px;background:var(--white);box-shadow:5px 5px 0 var(--black);min-width:180px}',
'.temp-label2{font-size:10px;font-weight:900;letter-spacing:.1em;opacity:.5}',
'.temp-val2{font-size:22px;font-weight:900;margin-top:2px}',
'.temp-hint{font-size:9px;font-weight:700;opacity:.45;margin-top:2px}',
'/* ===== 词汇表 ===== */',
'.two-col{display:grid;grid-template-columns:1fr 1fr;gap:20px;align-items:start}',
'@media(max-width:800px){.two-col{grid-template-columns:1fr}}',
'.vocab-grid{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:14px}',
'.vocab-chip{border:3px solid var(--black);padding:6px 8px;font-weight:900;font-size:14px;background:var(--white);box-shadow:3px 3px 0 var(--black);display:flex;flex-direction:column;align-items:center;min-width:36px;cursor:pointer;transition:all 80ms}',
'.vocab-chip:hover{background:var(--yellow);transform:translate(-1px,-1px);box-shadow:4px 4px 0 var(--black)}',
'.vocab-chip.vc-bos{background:var(--yellow)}',
'.vocab-chip.vc-sel{background:var(--black);color:var(--white)}',
'.vc-id{font-size:9px;font-weight:900;margin-top:2px;opacity:.5}',
'.tok-result{display:flex;flex-wrap:wrap;gap:6px;margin-top:12px;align-items:center}',
'.tok-card{border:3px solid var(--black);padding:8px 10px;background:var(--white);box-shadow:3px 3px 0 var(--black);text-align:center;animation:popIn .2s cubic-bezier(.34,1.56,.64,1)}',
'.tok-char{font-size:18px;font-weight:900;display:block;line-height:1}',
'.tok-id-lbl{font-size:9px;font-weight:900;opacity:.45;margin-top:3px}',
'.tok-arrow{font-size:18px;opacity:.2;font-weight:900}',
'/* ===== 嵌入矩阵 ===== */',
'.emb-scroll{overflow-x:auto;margin-bottom:8px}',
'.emb-row{display:flex;align-items:center;gap:1px;margin-bottom:1px}',
'.emb-lbl{font-size:9px;font-weight:900;width:34px;flex-shrink:0;text-align:right;padding-right:5px;opacity:.6;cursor:pointer}',
'.emb-lbl:hover{color:var(--red);opacity:1}',
'.emb-cell{width:13px;height:13px;border:1px solid rgba(0,0,0,.06);display:inline-block}',
'.emb-row.emb-hl .emb-lbl{color:var(--red);font-weight:900;opacity:1}',
'.emb-row.emb-hl .emb-cell{transform:scaleY(1.5);border:1px solid rgba(0,0,0,.25)}',
'.emb-val-strip{display:flex;gap:3px;flex-wrap:wrap;margin-top:8px;padding:10px;border:3px solid var(--black);background:var(--white);box-shadow:4px 4px 0 var(--black)}',
'.emb-v{width:28px;height:28px;border:2px solid rgba(0,0,0,.15);display:flex;align-items:center;justify-content:center;font-size:7px;font-weight:700;cursor:default}',
'/* ===== 种子说明 ===== */',
'.seed-note{border:3px solid var(--black);background:var(--violet);padding:8px 14px;font-size:11px;font-weight:700;margin-bottom:12px;box-shadow:3px 3px 0 var(--black);line-height:1.6}',
'/* 6个标签自适应宽度 */',
'.tab-btn{flex:1;padding:10px 3px;font-weight:900;font-size:11px;letter-spacing:.03em;border:none;border-right:3px solid var(--black);background:transparent;cursor:pointer;transition:background 80ms linear}'
].join('\n');


var VIZ_HTML = [
'<header class="hdr">',
'  <span class="hdr-badge">MICROGPT.JS</span>',
'  <span class="hdr-title">GPT 从零到一 · 六章教学可视化</span>',
'  <span class="hdr-sub">纯 JS · 无依赖 · 固定种子=42</span>',
'</header>',
'<nav class="tab-nav">',
'  <button class="tab-btn active" onclick="showTab(\'tok\')">① 词元化</button>',
'  <button class="tab-btn" onclick="showTab(\'emb\')">② 嵌入</button>',
'  <button class="tab-btn" onclick="showTab(\'attn\')">③ 注意力</button>',
'  <button class="tab-btn" onclick="showTab(\'grad\')">④ 损失与梯度</button>',
'  <button class="tab-btn" onclick="showTab(\'training\')">⑤ 训练</button>',
'  <button class="tab-btn" onclick="showTab(\'infer\')">⑥ 推理</button>',
'</nav>',
'',
'<!-- ①=== 词元化 ===-->',
'<div id="tab-tok" class="tab-panel active">',
'  <div class="tip">',
'    <strong>📘 第①步：词元化（Tokenization）</strong>　神经网络只能处理数字，不能直接处理字符。',
'    词元化把每个字符映射为一个整数 ID。',
'    本模型的词汇表只有 <strong>27 个词元</strong>：26 个小写字母 + 1 个特殊的 [BOS/EOS]（序列开始/结束符，ID=0）。',
'  </div>',
'  <div class="two-col">',
'    <div>',
'      <div class="section-title">词汇表（<span id="vocab-size">27</span> 个词元）</div>',
'      <div id="vocab-grid"></div>',
'      <div class="tip tip-white" style="font-size:11px">',
'        点击任意词元 → 在右侧嵌入矩阵中高亮其对应行。',
'        <br>[BOS] 是"句子开始/结束"的特殊符号，训练时作为输入的第一个词元，也是生成结束的信号。',
'      </div>',
'    </div>',
'    <div>',
'      <div class="section-title">演示：文本 → Token ID 序列</div>',
'      <div class="attn-input-row">',
'        <input id="tok-demo-input" class="attn-input" type="text" maxlength="14" placeholder="输入名字..." value="karisa">',
'        <button class="neo-btn dark" onclick="tokenizeDemo()">分词 ▶</button>',
'      </div>',
'      <div id="tok-demo-result" class="tok-result"></div>',
'      <div class="tip tip-violet" style="font-size:11px;margin-top:14px">',
'        <strong>为什么要加 [BOS]？</strong>',
'        模型需要知道"序列从哪里开始"。每个名字的前缀是 [BOS]，结尾也是 [BOS]（此时 BOS 充当 EOS）。',
'        训练时，输入 [BOS, k, a, r, i, s] → 模型依次预测 [k, a, r, i, s, a, BOS]。',
'      </div>',
'    </div>',
'  </div>',
'</div>',
'',
'<!-- ②=== 嵌入 ===-->',
'<div id="tab-emb" class="tab-panel">',
'  <div class="tip">',
'    <strong>📘 第②步：嵌入（Embedding）——把离散ID变成"有意义的向量"</strong><br>',
'    神经网络只能处理连续数字，不能直接用离散的 Token ID (0,1,2...)。',
'    <strong>词嵌入 wte</strong>：每个 Token → <strong>nEmb 维</strong>向量。初始值随机 [-16, 16)（避免梯度消失/爆炸），训练后自动学习相似关系（如 a/b 向量相近）。',
'    <strong>位置嵌入 wpe</strong>：每个位置 → nEmb 维向量。因为注意力本身不区分顺序（"I love you"和"you I love"对它一样），所以需要位置标记。',
'    <strong>最终输入 = wte + wpe</strong>，让同一字符在不同位置有不同表示。<span style="color:#999">← 点击下方词元可查看具体数值及解释</span>',
'  </div>',
'  <div class="two-col">',
'    <div>',
'      <div class="section-title">词嵌入矩阵 wte　[<span id="wte-shape"></span>]</div>',
'      <div class="tip tip-white" style="font-size:11px">',
'        每行 = 一个词元的嵌入向量（颜色越红=数值越大，越浅=越小/负）。',
'        <br>点击左侧词元标签 或 在"词元化"页点击词汇表 可高亮对应行。',
'      </div>',
'      <div id="emb-grid"></div>',
'    </div>',
'    <div>',
'      <div class="section-title">位置嵌入矩阵 wpe　[<span id="wpe-shape"></span>]（前8行）</div>',
'      <div class="tip tip-white" style="font-size:11px">',
'        每行 = 一个位置的嵌入向量。',
'        这让模型知道一个词元处于序列的第几个位置。',
'      </div>',
'      <div id="pos-emb-grid"></div>',
'      <div class="section-title" style="margin-top:16px">高亮词元的嵌入向量</div>',
'      <div id="emb-val-display" class="tip tip-white" style="font-size:11px">',
'        ← 点击词元标签可查看其具体嵌入数值',
'      </div>',
'    </div>',
'  </div>',
'</div>',
'',
'<!-- ③=== 注意力 ===-->',
'<div id="tab-attn" class="tab-panel">',
'  <div class="tip">',
'    <strong>📘 第③步：注意力机制——"让当前字符回头看上文"</strong><br>',
'    <strong>Q (Query)</strong>："我想知道什么？" <strong>K (Key)</strong>："我能提供什么？" <strong>V (Value)</strong>："具体信息"。',
'    计算流程：① 输入 x 投影得到 Q/K/V → ② Q·Kᵀ/√d 算相似度 → ③ Softmax 归一化（权重和=1） → ④ 加权求和 V。',
'    <strong>4个注意力头</strong>：每个头关注不同模式（相邻字符、长距离依赖、语义等），并行学习后拼接。',
'    <strong>因果掩码</strong>：位置 i 只能看 ≤i 的位置（不能看未来），保证生成时的自回归特性。<span style="color:#999">← 下方热力图深色=高权重</span>',
'  </div>',
'',
'  <div class="attn-input-row">',
'    <input id="attn-word-input" class="attn-input" type="text" maxlength="14" placeholder="输入一个名字..." value="karisa">',
'    <button class="neo-btn dark" onclick="runAttn()">运行 ▶</button>',
'    <span class="hint">输入任意名字，观察模型如何分配注意力</span>',
'  </div>',
'  <div class="attn-steps">',
'    <button class="astep active" onclick="selectAttnStep(0)"><span class="astep-num">①</span>输入词元</button>',
'    <button class="astep" onclick="selectAttnStep(1)"><span class="astep-num">②</span>Q·K·V 投影</button>',
'    <button class="astep" onclick="selectAttnStep(2)"><span class="astep-num">③</span>原始分数</button>',
'    <button class="astep" onclick="selectAttnStep(3)"><span class="astep-num">④</span>Softmax 热力图</button>',
'  </div>',
'  <div class="attn-step-box" id="attn-step-box">',
'    <div class="attn-step-title">点击上方步骤查看说明，或先输入名字点击"运行"</div>',
'  </div>',
'  <div class="head-tabs" id="head-tabs"></div>',
'  <div id="attn-heatmap-section">',
'    <div class="attn-grid">',
'      <div>',
'        <div class="section-title">注意力热力图 <span style="font-size:12px;opacity:.5">（行=Query 位置，列=Key 位置）</span></div>',
'        <div class="heatmap-outer"><div id="attn-heatmap"></div></div>',
'        <div class="heatmap-legend">',
'          <span>低</span><div class="legend-grad"></div><span>高</span>',
'          <span style="margin-left:auto" class="hint" id="attn-info">—</span>',
'        </div>',
'      </div>',
'      <div class="qkv-panel" id="qkv-panel">',
'        <div class="section-title" style="font-size:14px">Q · K · V</div>',
'      </div>',
'    </div>',
'  </div>',
'</div>',
'',
'<!-- ④=== 损失与梯度 ===-->',
'<div id="tab-grad" class="tab-panel">',
'  <div class="tip">',
'    <strong>📘 第④步：损失与梯度——"衡量预测好坏 + 指示如何改进"</strong><br>',
'    <strong>Loss（损失）</strong>：衡量模型预测与真实答案的差距。我们用<strong>交叉熵</strong>，因为这是<strong>分类问题</strong>（27个候选字符选一个）。',
'    公式：L = -log P(正确字符)。如果模型给正确答案概率p=0.7，Loss=-log(0.7)=0.36；如果p=0.01，Loss=4.6（很差）。',
'    <strong>梯度（Gradient）</strong>：∂L/∂θ，告诉每个参数"往哪调整才能让Loss变小"。负梯度=参数应增大，正梯度=应减小。',
'    <strong>归一化 (1/n)∑Loss</strong>：不同文本长度不同，除以n得到"平均每字符损失"，才能公平比较。<span style="color:#999">← 随机猜测Loss≈3.3，训练后降至1.8-2.0</span>',
'  </div>',
'',
'  <div class="two-col">',
'    <div>',
'      <div class="section-title">损失公式</div>',
'      <div class="tip tip-white" style="font-size:11px">',
'        <strong>交叉熵损失（Cross-Entropy Loss）：</strong>',
'        <br>L = − log P(正确词元)',
'        <br>= − log softmax(logits)[正确ID]',
'        <br><br>logits 是 lm_head 线性层的输出（每个词元一个分数）。',
'        Softmax 将分数转为概率（所有词元概率之和=1）。',
'        <br><br>模型输出概率越接近 1（对正确词元），Loss 越小（趋近 0）。',
'        模型完全随机（每词元概率=1/27），Loss ≈ log(27) ≈ <strong>3.30</strong>。',
'        <br>本次训练初始 Loss ≈ 3.24，最终降至约 <strong>1.8～2.0</strong>。',
'      </div>',
'      <div class="section-title" style="margin-top:16px">Adam 优化器</div>',
'      <div class="tip tip-white" style="font-size:11px">',
'        <strong>θ ← θ − lr · m̂ / (√v̂ + ε)</strong>',
'        <br>m̂：一阶动量（梯度的指数加权平均）',
'        <br>v̂：二阶动量（梯度平方的指数加权平均）',
'        <br>lr：学习率（线性衰减，从 0.01 → 0）',
'        <br><br>Adam 的优势：每个参数有自适应学习率，收敛更快更稳。',
'      </div>',
'    </div>',
'    <div>',
'      <div class="section-title">参数追踪器（选择训练步数后查看）</div>',
'      <div class="tip tip-violet" style="font-size:11px">',
'        <strong>梯度</strong>：反向传播后该参数行第一组 16 个值的导数',
'        <br><strong>参数值</strong>：当前权重值（Adam 更新后）',
'        <br>红=正，蓝=负，色深=幅度大。',
'        <br>步数滑块在"⑤训练"页，此处同步显示最近一步的参数状态。',
'      </div>',
'      <div class="param-tabs" id="param-tabs-grad"></div>',
'      <div class="vecs-row">',
'        <div class="vec-panel">',
'          <div class="vec-title">梯度 ∂L/∂θ</div>',
'          <div id="grad-bars-grad"></div>',
'        </div>',
'        <div class="vec-panel">',
'          <div class="vec-title">参数值 θ</div>',
'          <div id="val-bars-grad"></div>',
'        </div>',
'      </div>',
'    </div>',
'  </div>',
'</div>',
'',
'<!-- ⑤=== 训练 ===-->',
'<div id="tab-training" class="tab-panel">',
'  <div class="seed-note">',
'    💡 <strong>为什么损失曲线每次都一样？</strong>　因为 microgpt.js 使用固定随机种子 <code>_seed=42</code>，保证每次训练走相同路径，便于教学复现。<br>',
'    <strong>学习率 lr</strong>：参数更新步长 = lr * 梯度。初始 0.01→线性衰减→0。太大会震荡，太小会慢。Adam 优化器会自适应调整每个参数的实际步长（动量+二阶矩）。<br>',
'    <strong>有没有"模型文件"？</strong>　没有。viz.html 是一次训练的"录像"，权重只在运行时存在内存中，不会保存到磁盘。',
'  </div>',
'',
'  <div class="train-grid">',
'    <div class="info-col">',
'      <div class="neo-card"><div class="card-label">训练步数</div><div class="card-val" id="t-step">0</div></div>',
'      <div class="neo-card"><div class="card-label">当前训练词</div><div class="card-val sm" id="t-word">—</div></div>',
'      <div class="neo-card"><div class="card-label">损失值</div><div class="card-val" id="t-loss">—</div>',
'        <span class="loss-chg" id="loss-chg" style="display:none"></span>',
'      </div>',
'      <div class="neo-card"><div class="card-label">学习率</div><div class="card-val sm" id="t-lr">—</div></div>',
'    </div>',
'    <div>',
'      <div class="section-title">① 当前训练样本</div>',
'      <div class="train-sample-wrap">',
'        <div class="train-sample-lbl">上下文序列 → 预测下一词元</div>',
'        <div class="train-toks" id="train-toks">',
'          <span class="ttok t-bos">[开始]</span>',
'          <span class="ttok t-sep">→</span>',
'          <span class="ttok t-pred">?</span>',
'        </div>',
'      </div>',
'      <div class="section-title">② 梯度下降四步流程</div>',
'      <div class="pipe-row">',
'        <div class="pipe-stage" id="ps-fwd">前向传播<span class="ps-val">→</span></div>',
'        <div class="pipe-conn"><div class="pipe-fill" id="pf1"></div><div class="pipe-dot" id="pd1"></div></div>',
'        <div class="pipe-stage" id="ps-loss">损失<span class="ps-val" id="ps-loss-val">—</span></div>',
'        <div class="pipe-conn"><div class="pipe-fill" id="pf2"></div><div class="pipe-dot" id="pd2"></div></div>',
'        <div class="pipe-stage" id="ps-bwd">反向传播<span class="ps-val">←</span></div>',
'        <div class="pipe-conn"><div class="pipe-fill" id="pf3"></div><div class="pipe-dot" id="pd3"></div></div>',
'        <div class="pipe-stage" id="ps-upd">Adam 更新<span class="ps-val">θ′</span></div>',
'      </div>',
'      <div class="section-title">③ 损失曲线（固定种子=42，每次相同）</div>',
'      <div class="chart-wrap">',
'        <div class="chart-label">Loss 随训练步数变化（从 ~3.2 降至 ~1.8）</div>',
'        <svg id="loss-svg" viewBox="0 0 500 120" preserveAspectRatio="none">',
'          <rect width="500" height="120" fill="#fafafa"/>',
'          <polyline id="loss-polyline" class="loss-line" points=""/>',
'          <circle id="loss-cursor" class="loss-dot" cx="0" cy="0" r="4" fill="black"/>',
'        </svg>',
'      </div>',
'      <div class="ctrl-row">',
'        <button class="neo-btn" onclick="nudgeStep(-10)">«</button>',
'        <button class="neo-btn" onclick="nudgeStep(-1)">‹</button>',
'        <button class="neo-btn dark" id="play-btn" onclick="togglePlay()">▶ 播放</button>',
'        <button class="neo-btn" onclick="nudgeStep(1)">›</button>',
'        <button class="neo-btn" onclick="nudgeStep(10)">»</button>',
'        <button class="neo-btn red" onclick="resetPlay()">↺ 重置</button>',
'      </div>',
'      <div class="slider-wrap">',
'        <label>步数</label>',
'        <input type="range" id="step-slider" min="0" max="999" value="0" oninput="seekStep(+this.value)">',
'        <span class="step-readout" id="step-readout">0 / 1000</span>',
'      </div>',
'    </div>',
'    <div class="param-col">',
'      <div class="section-title">④ 参数追踪器</div>',
'      <div class="tip tip-violet" style="font-size:11px">',
'        红=正，蓝=负，色深=幅度大。',
'        <br>点击标签切换参数矩阵。',
'      </div>',
'      <div class="param-tabs" id="param-tabs"></div>',
'      <div class="vecs-row">',
'        <div class="vec-panel"><div class="vec-title">梯度 ∂L/∂θ</div><div id="grad-bars"></div></div>',
'        <div class="vec-panel"><div class="vec-title">参数值 θ</div><div id="val-bars"></div></div>',
'      </div>',
'    </div>',
'  </div>',
'</div>',
'',
'<!-- ⑥=== 推理 ===-->',
'<div id="tab-infer" class="tab-panel">',
'  <div class="tip">',
'    <strong>📘 第⑥步：推理生成（Autoregressive Inference）</strong>',
'    训练完成后，模型从 [BOS] 出发，每步通过前向传播得到 logits，',
'    Softmax + 温度采样得到下一个词元，追加到序列，如此循环直到采样到 [BOS]（此时充当 EOS）。',
'  </div>',
'  <div class="infer-ctrl">',
'    <div class="temp-card2">',
'      <div class="temp-label2">温度 Temperature</div>',
'      <input type="range" id="temp-slider" min="1" max="30" value="5" oninput="updateTemp(+this.value)" style="width:100%;margin:6px 0;accent-color:#000">',
'      <div class="temp-val2" id="temp-display">0.5</div>',
'      <div class="temp-hint">← 低温保守　高温随机 →</div>',
'    </div>',
'    <div style="display:flex;flex-direction:column;gap:8px">',
'      <button class="neo-btn dark" style="font-size:14px;padding:12px 24px" onclick="runInference()">⚡ 开始生成</button>',
'      <button class="neo-btn" onclick="clearSamples()">✕ 清除</button>',
'    </div>',
'    <div class="tip tip-white" style="font-size:11px;max-width:240px;margin-bottom:0">',
'      <strong>采样公式：</strong>p_i = softmax(logits/T)[i]<br>',
'      T→0：贪心（总选最高概率）<br>',
'      T=1：原始概率分布采样<br>',
'      T→∞：均匀随机',
'    </div>',
'  </div>',
'  <div class="section-title">生成词链（自回归逐词元追加）</div>',
'  <div class="token-chain" id="token-chain">',
'    <span class="ctok ct-bos">[BOS]</span>',
'    <span class="ctok ct-sep">→</span>',
'    <span style="font-size:12px;opacity:.4;font-weight:700">点击"开始生成"</span>',
'  </div>',
'  <div class="prob-wrap" id="prob-wrap" style="display:none">',
'    <div class="prob-title" id="prob-title">词元概率分布</div>',
'    <div id="prob-chart"></div>',
'  </div>',
'  <div class="section-title">已生成样本</div>',
'  <div class="samples-grid" id="samples-grid"></div>',
'</div>'
].join('\n');


var VIZ_JS = [
'// ============================================================',
'// UTILITIES',
'// ============================================================',
'function softmaxN(xs){const mx=xs.reduce((m,x)=>x>m?x:m,xs[0]);const ex=xs.map(x=>Math.exp(x-mx));const s=ex.reduce((a,b)=>a+b,0);return ex.map(e=>e/s);}',
'function rmsnormN(x){const ms=x.reduce((s,xi)=>s+xi*xi,0)/x.length;const sc=Math.pow(ms+1e-5,-0.5);return x.map(xi=>xi*sc);}',
'function linearN(x,w){return w.map(row=>row.reduce((s,wi,j)=>s+wi*x[j],0));}',
'function clamp(v,mn,mx){return Math.max(mn,Math.min(mx,v));}',
'function lerp(a,b,t){return a+(b-a)*t;}',
'function heatRgb(t){t=clamp(t,0,1);return `rgb(${Math.round(lerp(255,255,t))},${Math.round(lerp(253,107,t))},${Math.round(lerp(245,107,t))})`;}',
'function signedBarColor(v){return v>=0?"pos":"neg";}',
'',
'// ============================================================',
'// GPT FORWARD (plain numbers) — for Attention & Inference tabs',
'// ============================================================',
'function gptFull(tokens,S){',
'  const nL=S.nLayer,nH=S.nHead,hD=S.headDim,nE=S.nEmb;',
'  const layerKeys=[],layerVals=[];',
'  for(let li=0;li<nL;li++){layerKeys.push([]);layerVals.push([]);}',
'  const allAttn=[];  // [pos][layer][head] → softmax weights[]',
'  const allRawScores=[];  // [pos][layer][head] → raw scores before softmax',
'  const allLogits=[];',
'  for(let pos=0;pos<tokens.length;pos++){',
'    const tid=tokens[pos];',
'    let x=S.wte[tid].map((t,i)=>t+S.wpe[pos][i]);',
'    x=rmsnormN(x);',
'    const posAttn=[];',
'    const posRaw=[];',
'    for(let li=0;li<nL;li++){',
'      const xr=[...x];',
'      x=rmsnormN(x);',
'      const q=linearN(x,S["layer"+li+".attn_wq"]);',
'      const k=linearN(x,S["layer"+li+".attn_wk"]);',
'      const v=linearN(x,S["layer"+li+".attn_wv"]);',
'      layerKeys[li].push(k);',
'      layerVals[li].push(v);',
'      const ao=new Array(nE).fill(0);',
'      const layerAttn=[];',
'      const layerRaw=[];',
'      for(let h=0;h<nH;h++){',
'        const hs=h*hD;',
'        const lg=layerKeys[li].map(ki=>{let s=0;for(let j=0;j<hD;j++)s+=q[hs+j]*ki[hs+j];return s/Math.sqrt(hD);});',
'        const w=softmaxN(lg);',
'        layerAttn.push(w);',
'        layerRaw.push(lg);',
'        for(let j=0;j<hD;j++){let ws=0;for(let t=0;t<layerVals[li].length;t++)ws+=w[t]*layerVals[li][t][hs+j];ao[hs+j]=ws;}',
'      }',
'      posAttn.push(layerAttn);',
'      posRaw.push(layerRaw);',
'      const pr=linearN(ao,S["layer"+li+".attn_wo"]);',
'      x=pr.map((p,i)=>p+xr[i]);',
'      const xr2=[...x];',
'      x=rmsnormN(x);',
'      x=linearN(x,S["layer"+li+".mlp_fc1"]);',
'      x=x.map(xi=>Math.max(0,xi));',
'      x=linearN(x,S["layer"+li+".mlp_fc2"]);',
'      x=x.map((xi,i)=>xi+xr2[i]);',
'    }',
'    allAttn.push(posAttn);',
'    allRawScores.push(posRaw);',
'    allLogits.push(linearN(x,S.lm_head));',
'  }',
'  return {allAttn,allRawScores,allLogits};',
'}',
'',
'function tokenize(word){',
'  const toks=[SNAP.BOS];',
'  for(const ch of word.toLowerCase()){',
'    const idx=SNAP.uchars.indexOf(ch);',
'    if(idx>=0)toks.push(idx);',
'  }',
'  return toks;',
'}',
'',
'// ============================================================',
'// TAB SWITCHING',
'// ============================================================',
'// ============================================================',
'// 词元化 & 嵌入 初始化',
'// ============================================================',
'// === 词汇表 ===',
'(function initVocab(){',
'  const el=document.getElementById("vocab-grid");',
'  if(!el)return;',
'  const vszEl=document.getElementById("vocab-size");',
'  if(vszEl)vszEl.textContent=SNAP.uchars.length;',
'  SNAP.uchars.forEach((ch,id)=>{',
'    const chip=document.createElement("div");',
'    chip.className="vocab-chip"+(id===SNAP.BOS?" vc-bos":"");',
'    const lbl=ch===""?"[BOS]":ch.toUpperCase();',
'    chip.innerHTML=\'<span>\'+lbl+\'</span><span class="vc-id">ID:\'+id+\'</span>\';',
'    chip.onclick=()=>{',
'      document.querySelectorAll(".vocab-chip").forEach((c,i)=>c.classList.toggle("vc-sel",i===id));',
'      highlightEmbRow(id);',
'    };',
'    el.appendChild(chip);',
'  });',
'})();',
'',
'function tokenizeDemo(){',
'  const word=document.getElementById("tok-demo-input").value.trim();',
'  const el=document.getElementById("tok-demo-result");',
'  if(!el)return;',
'  const toks=[SNAP.BOS];',
'  for(const ch of word.toLowerCase()){const idx=SNAP.uchars.indexOf(ch);if(idx>=0)toks.push(idx);}',
'  toks.push(SNAP.BOS);',
'  let html="";',
'  toks.forEach((tid,i)=>{',
'    const ch=tid===SNAP.BOS?(i===0?"[BOS]":"[EOS]"):SNAP.uchars[tid].toUpperCase();',
'    html+=\'<div class="tok-card"><span class="tok-char">\'+ch+\'</span><span class="tok-id-lbl">ID \'+tid+\'</span></div>\';',
'    if(i<toks.length-1)html+=\'<span class="tok-arrow">→</span>\';',
'  });',
'  el.innerHTML=html;',
'}',
'tokenizeDemo();  // 初始化演示',
'',
'// === 嵌入矩阵 ===',
'(function initEmbedGrids(){',
'  const wteEl=document.getElementById("wte-shape");',
'  const wpeEl=document.getElementById("wpe-shape");',
'  if(wteEl)wteEl.textContent=SNAP.uchars.length+"×"+SNAP.nEmb;',
'  if(wpeEl)wpeEl.textContent="前8 × "+SNAP.nEmb;',
'  renderEmbGrid("emb-grid",SNAP.wte,SNAP.uchars.map((c,i)=>i===SNAP.BOS?"BOS":c.toUpperCase()));',
'  renderEmbGrid("pos-emb-grid",SNAP.wpe.slice(0,8),Array.from({length:8},(_,i)=>"pos"+i));',
'})();',
'',
'function renderEmbGrid(containerId,matrix,rowLabels){',
'  const el=document.getElementById(containerId);',
'  if(!el)return;',
'  let mn=Infinity,mx=-Infinity;',
'  matrix.forEach(row=>row.forEach(v=>{mn=Math.min(mn,v);mx=Math.max(mx,v);}));',
'  const range=Math.max(mx-mn,0.001);',
'  let html="<div class=\'emb-scroll\'>";',
'  matrix.forEach((row,ri)=>{',
'    html+=\'<div class="emb-row" id="er-\'+containerId+\'-\'+ri+\'">\';',
'    html+=\'<span class="emb-lbl" onclick="highlightEmbRow(\'+ri+\')">\'+rowLabels[ri]+\'</span>\';',
'    row.forEach(v=>{',
'      const t=(v-mn)/range;',
'      const bg=heatRgb(t);',
'      html+=\'<div class="emb-cell" style="background:\'+bg+\'" title="\'+v.toFixed(3)+\'"></div>\';',
'    });',
'    html+="</div>";',
'  });',
'  html+="</div>";',
'  el.innerHTML=html;',
'}',
'',
'function highlightEmbRow(tid){',
'  document.querySelectorAll(".emb-row").forEach(r=>r.classList.remove("emb-hl"));',
'  const rows=document.querySelectorAll(\'[id^="er-emb-grid-"]\');',
'  const target=document.getElementById("er-emb-grid-"+tid);',
'  if(target){',
'    target.classList.add("emb-hl");',
'    target.scrollIntoView({behavior:"smooth",block:"nearest"});',
'  }',
'  // 显示嵌入数值',
'  const dispEl=document.getElementById("emb-val-display");',
'  if(dispEl&&SNAP.wte[tid]){',
'    const vec=SNAP.wte[tid];',
'    let mn=Math.min(...vec),mx=Math.max(...vec);',
'    const range=Math.max(mx-mn,0.001);',
'    let html=\'<strong>wte[\'+tid+\']</strong>（\'+SNAP.uchars[tid].toUpperCase()+\'）dim=\'+vec.length+\' <span style="color:#999;font-size:10px">← 从随机 [-16,16) 初始化，训练后优化</span><br><div style="display:flex;gap:2px;flex-wrap:wrap;margin-top:6px">\';',
'    vec.forEach((v,i)=>{',
'      const t=(v-mn)/range;',
'      const bg=heatRgb(t);',
'      html+=\'<div class="emb-v" style="background:\'+bg+\'" title="[\'+i+\']=\'+v.toFixed(3)+\'">\'+v.toFixed(1)+\'</div>\';',
'    });',
'    html+=\'</div><div style="margin-top:10px;padding:8px;background:#FFF9E6;border:2px solid #000;font-size:11px;line-height:1.6">\';',
'    html+=\'<strong>💡 这些数字的含义：</strong><br>\';',
'    html+=\'• 每个数字代表词元"\'+SNAP.uchars[tid]+\'"在某个"特征维度"的强度<br>\';',
'    html+=\'• 相似的字符（如 a/b）会有相似的向量，模型通过训练自动学习<br>\';',
'    html+=\'• 范围 [\'+mn.toFixed(1)+\' ~ \'+mx.toFixed(1)+\']，颜色越红=数值越大\';',
'    html+="</div>";',
'    dispEl.innerHTML=html;',
'  }',
'}',
'',
'// ============================================================',
'// TAB SWITCHING — 更新为 6 个章节',
'// ============================================================',
'function showTab(name){',
'  document.querySelectorAll(".tab-panel").forEach(p=>p.classList.remove("active"));',
'  document.querySelectorAll(".tab-btn").forEach(b=>b.classList.remove("active"));',
'  document.getElementById("tab-"+name).classList.add("active");',
'  const idx={tok:0,emb:1,attn:2,grad:3,training:4,infer:5}[name]||0;',
'  document.querySelectorAll(".tab-btn")[idx].classList.add("active");',
'  // 注意力标签：首次打开自动运行',
'  if(name==="attn"&&!attnResult)setTimeout(runAttn,100);',
'  // 梯度标签：同步参数追踪器',
'  if(name==="grad")syncGradTab();',
'}',
'',
'// === 梯度标签同步 ===',
'var _gradTabInited=false;',
'var _selectedGradParam=0;',
'function syncGradTab(){',
'  if(!_gradTabInited){',
'    const el=document.getElementById("param-tabs-grad");',
'    if(el&&tp){',
'      tp.forEach((p,i)=>{',
'        const b=document.createElement("button");',
'        b.className="ptab"+(i===0?" active":"");',
'        b.textContent=p.label;',
'        b.onclick=(function(ii){return function(){',
'          _selectedGradParam=ii;',
'          document.querySelectorAll("#param-tabs-grad .ptab").forEach((x,j)=>x.classList.toggle("active",j===ii));',
'          const ph_idx=findNearestParam(lh[trainStep].s);',
'          renderVecBars("grad-bars-grad",ph[ph_idx]&&ph[ph_idx].p[tp[ii].id]?ph[ph_idx].p[tp[ii].id].g:new Array(16).fill(0),"#FF6B6B","#4488ff");',
'          renderVecBars("val-bars-grad",ph[ph_idx]&&ph[ph_idx].p[tp[ii].id]?ph[ph_idx].p[tp[ii].id].v:new Array(16).fill(0),"#FF6B6B","#4488ff");',
'        };})(i);',
'        el.appendChild(b);',
'      });',
'    }',
'    _gradTabInited=true;',
'  }',
'  const ph_idx=findNearestParam(lh[trainStep].s);',
'  if(ph&&tp){',
'    renderVecBars("grad-bars-grad",ph[ph_idx]&&ph[ph_idx].p[tp[_selectedGradParam].id]?ph[ph_idx].p[tp[_selectedGradParam].id].g:new Array(16).fill(0),"#FF6B6B","#4488ff");',
'    renderVecBars("val-bars-grad",ph[ph_idx]&&ph[ph_idx].p[tp[_selectedGradParam].id]?ph[ph_idx].p[tp[_selectedGradParam].id].v:new Array(16).fill(0),"#FF6B6B","#4488ff");',
'  }',
'}',
'',
'// ============================================================',
'// TRAINING PANEL',
'// ============================================================',
'let trainStep=0,isPlaying=false,playTimer=null,selectedParam=0;',
'const lh=TRACE.lossHistory;',
'const ph=TRACE.paramHistory;',
'const tp=TRACE.trackedParams;',
'',
'// Build param tabs',
'(function(){',
'  const el=document.getElementById("param-tabs");',
'  tp.forEach((p,i)=>{',
'    const b=document.createElement("button");',
'    b.className="ptab"+(i===0?" active":"");',
'    b.textContent=p.label;',
'    b.onclick=()=>selectParam(i);',
'    el.appendChild(b);',
'  });',
'  renderBars(0,0);',
'})();',
'',
'// Build loss chart',
'(function(){',
'  const pts=lh.map((r,i)=>{',
'    const x=2+i/lh.length*496;',
'    return {x,l:r.l};',
'  });',
'  const maxL=Math.max(...pts.map(p=>p.l));',
'  const minL=Math.min(...pts.map(p=>p.l));',
'  const range=Math.max(maxL-minL,0.01);',
'  window._chartPts=pts.map(p=>({x:p.x,y:4+(1-(p.l-minL)/range)*112}));',
'  window._chartMin=minL;window._chartMax=maxL;',
'  document.getElementById("loss-polyline").setAttribute("points",',
'    window._chartPts.map(p=>p.x+","+p.y).join(" "));',
'  document.getElementById("step-slider").max=lh.length-1;',
'  renderTrainStep(0);',
'})();',
'',
'function renderTrainStep(idx){',
'  idx=clamp(idx,0,lh.length-1);',
'  const prevLoss=trainStep!==idx&&lh[trainStep]?lh[trainStep].l:null;',
'  trainStep=idx;',
'  document.getElementById("step-slider").value=idx;',
'  document.getElementById("step-readout").textContent=(lh[idx].s)+" / "+lh.length;',
'  document.getElementById("t-step").textContent=lh[idx].s;',
'  document.getElementById("t-word").textContent=lh[idx].w.toUpperCase();',
'  document.getElementById("t-loss").textContent=lh[idx].l.toFixed(4);',
'  document.getElementById("t-lr").textContent=lh[idx].lr.toFixed(6);',
'  renderTrainExample(lh[idx].w);',
'  animatePipeline(lh[idx].l, prevLoss);',
'  // move cursor on chart',
'  if(window._chartPts&&window._chartPts[idx]){',
'    const p=window._chartPts[idx];',
'    document.getElementById("loss-cursor").setAttribute("cx",p.x);',
'    document.getElementById("loss-cursor").setAttribute("cy",p.y);',
'  }',
'  // find nearest param history entry',
'  const ph_idx=findNearestParam(lh[idx].s);',
'  renderBars(ph_idx,selectedParam);',
'}',
'',
'// === 训练词元芯片 ===',
'function renderTrainExample(word){',
'  const el=document.getElementById("train-toks");',
'  if(!el)return;',
'  const chars=word.toLowerCase().split("").filter(c=>SNAP.uchars.indexOf(c)>=0);',
'  let html=\'<span class="ttok t-bos">[开始]</span>\';',
'  chars.forEach(ch=>{',
'    html+=\'<span class="ttok t-sep">→</span>\';',
'    html+=\'<span class="ttok t-ctx">\'+ch.toUpperCase()+\'</span>\';',
'  });',
'  html+=\'<span class="ttok t-sep">→</span><span class="ttok t-pred" title="模型正在预测此位置">?</span>\';',
'  el.innerHTML=html;',
'}',
'',
'// === 梯度下降流程管道动画 ===',
'var _prevPipeClass="";',
'function animatePipeline(loss,prevLoss){',
'  // 颜色映射：loss > 3 热红，> 2.5 橙，> 2 黄，≤ 2 绿',
'  const cls=loss>3?"ps-hot":loss>2.5?"ps-warm":loss>2?"ps-ok":"ps-cool";',
'  const lossNode=document.getElementById("ps-loss");',
'  const lossVal=document.getElementById("ps-loss-val");',
'  if(lossNode){',
'    lossNode.className="pipe-stage "+cls;',
'    if(lossVal)lossVal.textContent=loss.toFixed(3);',
'  }',
'  // 变化指示器',
'  const chgEl=document.getElementById("loss-chg");',
'  if(chgEl&&prevLoss!==null){',
'    const d=loss-prevLoss;',
'    if(Math.abs(d)>0.001){',
'      chgEl.style.display="block";',
'      chgEl.textContent=(d>0?"▲ +":"▼ ")+d.toFixed(3);',
'      chgEl.className="loss-chg "+(d>0?"lc-up":"lc-dn");',
'    }',
'  }',
'  // 流程动画',
'  ["pf1","pf2","pf3"].forEach((id,i)=>{',
'    const el=document.getElementById(id);',
'    if(!el)return;',
'    el.className="pipe-fill";',
'    setTimeout(()=>el.className="pipe-fill go",i*180);',
'  });',
'  ["pd1","pd2","pd3"].forEach((id,i)=>{',
'    const el=document.getElementById(id);',
'    if(!el)return;',
'    el.className="pipe-dot";',
'    setTimeout(()=>el.className="pipe-dot go",i*180);',
'  });',
'}',
'',
'function findNearestParam(step){',
'  let best=0,bestDiff=Infinity;',
'  for(let i=0;i<ph.length;i++){',
'    const d=Math.abs(ph[i].s-step);',
'    if(d<bestDiff){bestDiff=d;best=i;}',
'  }',
'  return best;',
'}',
'',
'function renderBars(phIdx,paramIdx){',
'  const pid=tp[paramIdx].id;',
'  const rec=ph[phIdx];',
'  const gv=rec&&rec.p[pid]?rec.p[pid].g:new Array(16).fill(0);',
'  const vv=rec&&rec.p[pid]?rec.p[pid].v:new Array(16).fill(0);',
'  renderVecBars("grad-bars",gv,"#FF6B6B","#4488ff");',
'  renderVecBars("val-bars",vv,"#FF6B6B","#4488ff");',
'}',
'',
'function renderVecBars(containerId,vals,posColor,negColor){',
'  const el=document.getElementById(containerId);',
'  el.innerHTML="";',
'  const maxAbs=vals.reduce((m,v)=>Math.max(m,Math.abs(v)),0.001);',
'  vals.forEach((v,i)=>{',
'    const pct=Math.min(1,Math.abs(v)/maxAbs)*50;',
'    const item=document.createElement("div");',
'    item.className="bar-item";',
'    const sign=v>=0?"pos":"neg";',
'    const fillStyle=sign==="pos"',
'      ?"left:50%;width:"+pct+"%"',
'      :"right:50%;width:"+pct+"%";',
'    const col=sign==="pos"?posColor:negColor;',
'    item.innerHTML=`<span class="bar-idx">${i}</span><span class="bar-track"><span class="bar-fill" style="${fillStyle};background:${col}"></span></span><span class="bar-num">${v.toFixed(3)}</span>`;',
'    el.appendChild(item);',
'  });',
'}',
'',
'function selectParam(i){',
'  selectedParam=i;',
'  document.querySelectorAll(".ptab").forEach((b,idx)=>b.classList.toggle("active",idx===i));',
'  const ph_idx=findNearestParam(lh[trainStep].s);',
'  renderBars(ph_idx,i);',
'}',
'',
'function seekStep(v){',
'  stopPlay();',
'  renderTrainStep(v);',
'}',
'',
'function nudgeStep(d){',
'  stopPlay();',
'  renderTrainStep(trainStep+d);',
'}',
'',
'function togglePlay(){',
'  if(isPlaying)stopPlay();',
'  else startPlay();',
'}',
'',
'function startPlay(){',
'  if(trainStep>=lh.length-1)trainStep=0;',
'  isPlaying=true;',
'  document.getElementById("play-btn").textContent="⏸ 暂停";',
'  document.getElementById("play-btn").classList.add("red");',
'  tick();',
'}',
'',
'function stopPlay(){',
'  isPlaying=false;',
'  if(playTimer){clearTimeout(playTimer);playTimer=null;}',
'  document.getElementById("play-btn").textContent="▶ 播放";',
'  document.getElementById("play-btn").classList.remove("red");',
'}',
'',
'function tick(){',
'  if(!isPlaying)return;',
'  const next=trainStep+1;',
'  if(next>=lh.length){stopPlay();return;}',
'  renderTrainStep(next);',
'  playTimer=setTimeout(tick,60);',
'}',
'',
'function resetPlay(){stopPlay();renderTrainStep(0);}',
'',
'// ============================================================',
'// ATTENTION PANEL',
'// ============================================================',
'let attnResult=null,selectedHead=0,selectedRow=0,attnStepIdx=0;',
'',
'// === 注意力分步导览 ===',
'function selectAttnStep(idx){',
'  attnStepIdx=idx;',
'  document.querySelectorAll(".astep").forEach((b,i)=>b.classList.toggle("active",i===idx));',
'  renderAttnStepBox();',
'}',
'',
'function renderAttnStepBox(){',
'  const box=document.getElementById("attn-step-box");',
'  if(!box)return;',
'  const heatSec=document.getElementById("attn-heatmap-section");',
'  if(attnStepIdx===0){',
'    // ① 输入词元',
'    box.innerHTML=\'<div class="attn-step-title">① 输入词元 — 每个字母被映射为一个 Token ID，再经过 Embedding 变成向量</div>\';',
'    if(attnResult){',
'      const toks=attnResult.tokens;',
'      let html=\'<div class="train-toks" style="margin-top:6px">\';',
'      toks.forEach((t,i)=>{',
'        const lbl=t===SNAP.BOS?"[开始]":SNAP.uchars[t].toUpperCase();',
'        const bg=t===SNAP.BOS?"background:var(--yellow)":"background:var(--white)";',
'        html+=`<span class="ttok t-ctx" style="${bg}" title="ID=${t} 位置=${i}">${lbl}</span>`;',
'        if(i<toks.length-1)html+=\'<span class="ttok t-sep">→</span>\';',
'      });',
'      html+=\'</div>\';',
'      html+=`<div style="margin-top:12px;font-size:11px;font-weight:700;opacity:.6">共 ${toks.length} 个词元 · 词嵌入维度 nEmb=${SNAP.nEmb} · 位置编码维度=${SNAP.nEmb}</div>`;',
'      box.innerHTML+=html;',
'    } else { box.innerHTML+=\'<div style="opacity:.5;font-size:12px;font-weight:700;margin-top:8px">请先输入名字并点击"运行"</div>\'; }',
'    if(heatSec)heatSec.style.display="none";',
'  } else if(attnStepIdx===1){',
'    // ② Q/K/V 投影',
'    box.innerHTML=\'<div class="attn-step-title">② Q·K·V 线性投影 — x 分别乘以 Wq、Wk、Wv 矩阵，得到查询、键、值向量</div>\';',
'    if(attnResult){',
'      box.innerHTML+=\'<div style="font-size:11px;font-weight:700;opacity:.6;margin-bottom:8px">点击热力图中的某行，右侧会显示对应位置的 Q/K/V 向量（颜色 = 数值大小）</div>\';',
'    }',
'    if(heatSec)heatSec.style.display="block";',
'    renderQKV(selectedRow,attnResult?attnResult.tokens:[]);',
'  } else if(attnStepIdx===2){',
'    // ③ 原始注意力分数',
'    box.innerHTML=\'<div class="attn-step-title">③ 原始注意力分数 = Q · Kᵀ / √d — 值越大说明这对 (Query, Key) 越"相关"；上三角被掩码为 −∞</div>\';',
'    if(attnResult){',
'      renderRawScoreMatrix(box);',
'    } else { box.innerHTML+=\'<div style="opacity:.5;font-size:12px;font-weight:700;margin-top:8px">请先输入名字并点击"运行"</div>\'; }',
'    if(heatSec)heatSec.style.display="none";',
'  } else {',
'    // ④ Softmax 权重热力图',
'    box.innerHTML=\'<div class="attn-step-title">④ Softmax 归一化 → 注意力权重热力图 — 每行之和为 1，越红表示越"关注"该位置；对角线以上为 0（因果掩码）</div>\';',
'    if(heatSec)heatSec.style.display="block";',
'    if(attnResult)renderHeatmap();',
'  }',
'}',
'',
'function renderRawScoreMatrix(container){',
'  if(!attnResult||!attnResult.allRawScores)return;',
'  const tokens=attnResult.tokens;',
'  const nPos=tokens.length;',
'  const rawHead=attnResult.allRawScores.map(posRaw=>posRaw[0]?posRaw[0][selectedHead]:null);',
'  // Find min/max for color normalization',
'  let mn=Infinity,mx=-Infinity;',
'  rawHead.forEach((row,i)=>{if(!row)return;row.forEach((v,j)=>{if(j<=i){mn=Math.min(mn,v);mx=Math.max(mx,v);}});});',
'  const range=Math.max(mx-mn,0.001);',
'  let html=\'<div style="overflow-x:auto;margin-top:8px"><div class="score-grid">\';',
'  // Header row',
'  html+=\'<div class="score-row"><div class="sc sc-hd" style="min-width:46px">Q↓ K→</div>\';',
'  tokens.forEach((t,j)=>{',
'    const lbl=t===SNAP.BOS?"开始":SNAP.uchars[t].toUpperCase();',
'    html+=`<div class="sc sc-hd">${lbl}</div>`;',
'  });',
'  html+=\'</div>\';',
'  // Data rows',
'  for(let i=0;i<nPos;i++){',
'    const rowLbl=tokens[i]===SNAP.BOS?"开始":SNAP.uchars[tokens[i]].toUpperCase();',
'    html+=`<div class="score-row"><div class="sc sc-hd">${rowLbl}</div>`;',
'    for(let j=0;j<nPos;j++){',
'      if(j>i){',
'        html+=\'<div class="sc" style="background:#f0f0f0;color:#ccc">—</div>\';',
'      } else {',
'        const v=rawHead[i]&&rawHead[i][j]!==undefined?rawHead[i][j]:0;',
'        const t=(v-mn)/range;',
'        const bg=heatRgb(t*1.2);',
'        html+=`<div class="sc" style="background:${bg}" title="${v.toFixed(3)}">${v.toFixed(1)}</div>`;',
'      }',
'    }',
'    html+=\'</div>\';',
'  }',
'  html+=\'</div></div>\';',
'  html+=`<div style="font-size:10px;font-weight:700;opacity:.5;margin-top:6px">头 ${selectedHead} · 未归一化分数（交叉熵前）· 颜色越深=分数越高=注意力越强</div>`;',
'  container.innerHTML+=html;',
'}',
'',
'// Build head tabs',
'(function(){',
'  const el=document.getElementById("head-tabs");',
'  for(let h=0;h<SNAP.nHead;h++){',
'    const b=document.createElement("button");',
'    b.className="htab"+(h===0?" active":"");',
'    b.textContent="注意力头 "+h;',
'    b.onclick=((hh)=>()=>{selectedHead=hh;document.querySelectorAll(".htab").forEach((x,i)=>x.classList.toggle("active",i===hh));if(attnResult){renderAttnStepBox();renderHeatmap();};})(h);',
'    el.appendChild(b);',
'  }',
'})();',
'',
'function runAttn(){',
'  const word=document.getElementById("attn-word-input").value.trim();',
'  if(!word)return;',
'  const tokens=tokenize(word);',
'  const res=gptFull(tokens,SNAP);',
'  attnResult=res;',
'  attnResult.tokens=tokens;',
'  attnResult.word=word;',
'  attnResult.allRawScores=res.allRawScores;',
'  selectedRow=0;',
'  renderAttnStepBox();',
'  renderTokenDisplay(tokens);',
'  renderHeatmap();',
'  renderQKV(0,tokens);',
'}',
'',
'function renderTokenDisplay(tokens){',
'  const el=document.getElementById("tok-display");',
'  if(!el)return;',
'  el.innerHTML="";',
'  tokens.forEach((t,i)=>{',
'    const ch=document.createElement("div");',
'    ch.className="tok-chip"+(t===SNAP.BOS?" bos":"")+(i===selectedRow?" active-query":"");',
'    ch.textContent=t===SNAP.BOS?"[开始]":SNAP.uchars[t].toUpperCase();',
'    ch.title="位置 "+i+" | 词元ID "+t;',
'    ch.onclick=(function(ii){return function(){selectedRow=ii;renderTokenDisplay(tokens);renderAttnStepBox();renderHeatmap();renderQKV(ii,tokens);};})(i);',
'    el.appendChild(ch);',
'  });',
'}',
'',
'function renderHeatmap(){',
'  if(!attnResult)return;',
'  const tokens=attnResult.tokens;',
'  const nPos=tokens.length;',
'  const el=document.getElementById("attn-heatmap");',
'  el.innerHTML="";',
'  const table=document.createElement("table");',
'  table.className="heatmap-table";',
'  // Header row',
'  const thead=document.createElement("tr");',
'  thead.appendChild(Object.assign(document.createElement("th"),{textContent:"查询↓ 键→"}));',
'  for(let j=0;j<nPos;j++){',
'    const t=tokens[j];',
'    const th=document.createElement("th");',
'    th.textContent=t===SNAP.BOS?"开始":SNAP.uchars[t].toUpperCase();',
'    thead.appendChild(th);',
'  }',
'  table.appendChild(thead);',
'  // Data rows',
'  for(let i=0;i<nPos;i++){',
'    const tr=document.createElement("tr");',
'    const lbl=document.createElement("td");',
'    const tok=tokens[i];',
'    lbl.textContent=tok===SNAP.BOS?"开始":SNAP.uchars[tok].toUpperCase();',
'    lbl.style.background="var(--yellow)";',
'    lbl.style.fontWeight="900";',
'    tr.appendChild(lbl);',
'    const ws=attnResult.allAttn[i][0][selectedHead];',
'    for(let j=0;j<nPos;j++){',
'      const td=document.createElement("td");',
'      const w=j<ws.length?ws[j]:0;',
'      const bg=heatRgb(w*1.5);',
'      const div=document.createElement("div");',
'      div.className="hmcell";',
'      div.style.background=bg;',
'      div.textContent=w.toFixed(2);',
'      div.title="位置 "+i+" 关注位置 "+j+": "+w.toFixed(4);',
'      td.appendChild(div);',
'      tr.appendChild(td);',
'    }',
'    tr.onclick=(function(ii){return function(){selectedRow=ii;renderTokenDisplay(tokens);renderHeatmap();renderQKV(ii,tokens);};})(i);',
'    if(i===selectedRow)tr.style.outline="3px solid var(--black)";',
'    table.appendChild(tr);',
'  }',
'  el.appendChild(table);',
'  document.getElementById("attn-info").textContent="查询行 "+selectedRow+" | 注意力头 "+selectedHead+" | 词: "+attnResult.word;',
'}',
'',
'function renderQKV(posIdx,tokens){',
'  const el=document.getElementById("qkv-panel");',
'  el.innerHTML=\'<div class="section-title" style="font-size:14px">Q · K · V　<span style="font-size:11px;opacity:.5">位置 \'+posIdx+\'</span><div class="tip tip-violet" style="font-size:11px;margin-top:8px"><strong>Q（查询）</strong>：当前位置想找什么<br><strong>K（键）</strong>：其他位置有什么<br><strong>V（值）</strong>：传递的实际信息<br>颜色越红＝数值越大，越浅＝越小</div></div>\';',
'  if(!SNAP||!tokens)return;',
'  const S=SNAP;',
'  const tid=tokens[posIdx];',
'  let x=S.wte[tid].map((t,i)=>t+S.wpe[posIdx][i]);',
'  x=rmsnormN(x);',
'  x=rmsnormN(x);',
'  const q=linearN(x,S["layer0.attn_wq"]);',
'  const k=linearN(x,S["layer0.attn_wk"]);',
'  const v=linearN(x,S["layer0.attn_wv"]);',
'  [["Q",q],["K",k],["V",v]].forEach(([label,vec])=>{',
'    const block=document.createElement("div");',
'    block.className="qkv-block";',
'    block.innerHTML=\'<div class="qkv-title">\'+label+\'</div>\';',
'    const row=document.createElement("div");',
'    row.className="qkv-row";',
'    const mx=Math.max(...vec.map(Math.abs),0.001);',
'    vec.forEach(v=>{',
'      const cell=document.createElement("div");',
'      cell.className="qkv-cell";',
'      const t=clamp((v/mx+1)/2,0,1);',
'      cell.style.background=heatRgb(t);',
'      cell.title=v.toFixed(4);',
'      cell.textContent=v>=0?"+":"-";',
'      row.appendChild(cell);',
'    });',
'    block.appendChild(row);',
'    el.appendChild(block);',
'  });',
'}',
'',
'// 注意力自动运行已移至 showTab()',
'',
'// ============================================================',
'// INFERENCE PANEL',
'// ============================================================',
'let inferTemp=0.5;',
'let inferSamples=[];',
'// Pre-fill with training results',
'(function(){',
'  if(TRACE.inferenceResults&&TRACE.inferenceResults.length){',
'    TRACE.inferenceResults.forEach(name=>{',
'      const chip=document.createElement("div");',
'      chip.className="sample-chip";',
'      chip.textContent=name.toUpperCase();',
'      document.getElementById("samples-grid").appendChild(chip);',
'    });',
'  }',
'})();',
'',
'function updateTemp(v){inferTemp=v/10;document.getElementById("temp-display").textContent=inferTemp.toFixed(1);}',
'',
'function runInference(){',
'  const S=SNAP;',
'  const chainEl=document.getElementById("token-chain");',
'  chainEl.innerHTML=\'<span class="ctok ct-bos">[开始]</span>\';',
'  document.getElementById("prob-wrap").style.display="block";',
'  ',
'  let token=S.BOS;',
'  const layerKeys=[],layerVals=[];',
'  for(let li=0;li<S.nLayer;li++){layerKeys.push([]);layerVals.push([]);}',
'  const toks=[];',
'  ',
'  function step(pos){',
'    const tid=token;',
'    let x=S.wte[tid].map((t,i)=>t+S.wpe[pos][i]);',
'    x=rmsnormN(x);',
'    for(let li=0;li<S.nLayer;li++){',
'      const xr=[...x];',
'      x=rmsnormN(x);',
'      const q=linearN(x,S["layer"+li+".attn_wq"]);',
'      const k=linearN(x,S["layer"+li+".attn_wk"]);',
'      const v=linearN(x,S["layer"+li+".attn_wv"]);',
'      layerKeys[li].push(k);',
'      layerVals[li].push(v);',
'      const ao=new Array(S.nEmb).fill(0);',
'      for(let h=0;h<S.nHead;h++){',
'        const hs=h*S.headDim;',
'        const lg=layerKeys[li].map(ki=>{let s=0;for(let j=0;j<S.headDim;j++)s+=q[hs+j]*ki[hs+j];return s/Math.sqrt(S.headDim);});',
'        const w=softmaxN(lg);',
'        for(let j=0;j<S.headDim;j++){let ws=0;for(let t=0;t<layerVals[li].length;t++)ws+=w[t]*layerVals[li][t][hs+j];ao[hs+j]=ws;}',
'      }',
'      const pr=linearN(ao,S["layer"+li+".attn_wo"]);',
'      x=pr.map((p,i)=>p+xr[i]);',
'      const xr2=[...x];',
'      x=rmsnormN(x);',
'      x=linearN(x,S["layer"+li+".mlp_fc1"]);',
'      x=x.map(xi=>Math.max(0,xi));',
'      x=linearN(x,S["layer"+li+".mlp_fc2"]);',
'      x=x.map((xi,i)=>xi+xr2[i]);',
'    }',
'    const logits=linearN(x,S.lm_head);',
'    const mx=logits.reduce((m,v)=>v>m?v:m,logits[0]);',
'    const ex=logits.map(v=>Math.exp((v-mx)/inferTemp));',
'    const sum=ex.reduce((a,b)=>a+b,0);',
'    const probs=ex.map(e=>e/sum);',
'    // sample',
'    let r=Math.random(),acc=0,next=S.BOS;',
'    for(let i=0;i<probs.length;i++){acc+=probs[i];if(r<acc){next=i;break;}}',
'    renderProbs(probs,next);',
'    if(next===S.BOS||pos>=S.blockSize-1)return null;',
'    toks.push(next);',
'    // 追加到词链',
'    const sep=document.createElement("span");',
'    sep.className="ctok ct-sep";sep.textContent="→";',
'    const chip=document.createElement("span");',
'    chip.className="ctok ct-new";',
'    chip.textContent=S.uchars[next].toUpperCase();',
'    chip.title="位置 "+pos+" · Token ID "+next+" · 概率 "+(probs[next]*100).toFixed(1)+"%";',
'    chainEl.appendChild(sep);',
'    chainEl.appendChild(chip);',
'    setTimeout(()=>chip.classList.replace("ct-new","ct-old"),350);',
'    token=next;',
'    return next;',
'  }',
'  ',
'  let pos=0;',
'  function doStep(){',
'    const r=step(pos);',
'    pos++;',
'    if(r!==null&&pos<S.blockSize)setTimeout(doStep,200);',
'    else{',
'      const name=toks.map(t=>S.uchars[t]).join("");',
'      if(name){',
'        const chip=document.createElement("div");',
'        chip.className="sample-chip new";',
'        chip.textContent=name.toUpperCase();',
'        document.getElementById("samples-grid").prepend(chip);',
'        setTimeout(()=>chip.classList.remove("new"),500);',
'        inferSamples.push(name);',
'      }',
'    }',
'  }',
'  doStep();',
'}',
'',
'function renderProbs(probs,sampled){',
'  const el=document.getElementById("prob-chart");',
'  el.innerHTML="";',
'  const S=SNAP;',
'  // top 10 by probability',
'  const indexed=probs.map((p,i)=>({p,i})).sort((a,b)=>b.p-a.p).slice(0,10);',
'  document.getElementById("prob-title").textContent="词元概率分布　（已采样: "+(sampled===S.BOS?"[结束]":S.uchars[sampled].toUpperCase())+")";',
'  indexed.forEach(({p,i})=>{',
'    const label=i===S.BOS?"[结束]":S.uchars[i].toUpperCase();',
'    const item=document.createElement("div");',
'    item.className="prob-bar-item";',
'    const fillBg=i===sampled?"var(--red)":"var(--black)";',
'    item.innerHTML=`<span class="prob-char">${label}</span><span class="prob-track"><span class="prob-fill" style="width:${(p*100).toFixed(1)}%;background:${fillBg}"></span></span><span class="prob-num">${(p*100).toFixed(1)}%</span>`;',
'    el.appendChild(item);',
'  });',
'}',
'',
'function clearSamples(){',
'  const chain=document.getElementById("token-chain");',
'  if(chain)chain.innerHTML=\'<span class="ctok ct-bos">[开始]</span><span class="ctok ct-sep">→</span><span style="font-size:12px;opacity:.4;font-weight:700">点击"开始生成"查看词元逐步追加过程</span>\';',
'  document.getElementById("samples-grid").innerHTML="";',
'  document.getElementById("prob-chart").innerHTML="";',
'  document.getElementById("prob-wrap").style.display="none";',
'  inferSamples=[];',
'}'
].join('\n');


// ============================================================
// TRACE 模式：生成 viz.html（在 VIZ_* 变量赋值之后调用）
// ============================================================
if (doTrace) {
    console.log('\n生成可视化文件...');

    // 导出 snapshot（所有权重）
    var snapshot = {
        uchars: uchars,
        BOS: BOS,
        nEmb: nEmb,
        nHead: nHead,
        headDim: headDim,
        nLayer: nLayer,
        blockSize: blockSize,
        trackedParams: trackedMatrices
    };
    for (var k in state) {
        snapshot[k] = state[k].map(function(row) {
            return row.map(function(v) { return r4(v.data); });
        });
    }

    var tracePayload = {
        numSteps: steps,
        lossHistory: lossHistory,
        paramHistory: paramHistory,
        trackedParams: trackedMatrices,
        inferenceResults: inferenceResults
    };

    var vizHtml = buildVizHtml(tracePayload, snapshot);
    fs.writeFileSync(path.join(__dirname, 'viz.html'), vizHtml, 'utf8');
    console.log('\u2713 可视化文件已生成：viz.html  直接用浏览器打开即可。');
}
