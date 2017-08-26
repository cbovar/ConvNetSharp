using System;
using System.Collections.Generic;

namespace ConvnetSharpOLD
{
    [Serializable]
    public class Trainer
    {
        Net net;
        Options options;

        double learningRate;
        double l1_decay;
        double l2_decay;
        public double batchSize;
        string method; 

        double momentum;
        double ro;
        double eps;
        double beta1;
        double beta2;

        double k = 0; // iteration counter
        List<double[]> gsum; // last iteration gradients (used for momentum calculations)
        List<double[]> xsum; // used in adam or adadelta
        public bool regression;

        Util util;

        public Trainer(Net net, Options options)
        {
            this.util = new Util();
            this.net = net;

            this.options = options;
            this.learningRate = options.learningRate != double.MinValue ? options.learningRate : 0.01;
            this.l1_decay = options.l1_decay != double.MinValue ? options.l1_decay : 0.0;
            this.l2_decay = options.l2_decay != double.MinValue ? options.l2_decay : 0.0;
            this.batchSize = options.batchSize != int.MinValue ? options.batchSize : 1;

            // methods: sgd/adam/adagrad/adadelta/windowgrad/netsterov
            this.method = string.IsNullOrEmpty(options.method) ? "sgd" : options.method; 

            this.momentum = options.momentum != double.MinValue ? options.momentum : 0.9;
            this.ro = options.ro != double.MinValue ? options.ro : 0.95; // used in adadelta
            this.eps = options.eps != double.MinValue ? options.eps : 1e-8; // used in adam or adadelta
            this.beta1 = options.beta1 != double.MinValue ? options.beta1 : 0.9; // used in adam
            this.beta2 = options.beta2 != double.MinValue ? options.beta2 : 0.999; // used in adam

            this.gsum = new List<double[]>();
            this.xsum = new List<double[]>();

            // check if regression is expected 
            if (this.net.layers[this.net.layers.Count - 1].type == "regression")
                this.regression = true;
            else
                this.regression = false;
        }

        public Dictionary<string, string> train(Volume x, object y)
        {
            var start = new DateTime();
            this.net.forward(x, true); // also set the flag that lets the net know we're just training
            var end = new DateTime();
            var fwd_time = end - start;

            start = new DateTime();
            var cost_loss = this.net.backward(y);
            var l2_decay_loss = 0.0;
            var l1_decay_loss = 0.0;
            end = new DateTime();
            var bwd_time = end - start;

            //if (this.regression && y.GetType().Equals(typeof(Array)) == false)
                //Console.WriteLine("Warning: a regression net requires an array as training output vector.");

            this.k++;
            if (this.k % this.batchSize == 0)
            {
                var pglist = this.net.getParamsAndGrads();

                // initialize lists for accumulators. Will only be done once on first iteration
                if (this.gsum.Count == 0 && (this.method != "sgd" || this.momentum > 0.0))
                {
                    // only vanilla sgd doesnt need either lists
                    // momentum needs gsum
                    // adagrad needs gsum
                    // adam and adadelta needs gsum and xsum
                    for (var i = 0; i < pglist.Length; i++)
                    {
                        this.gsum.Add(util.zeros(pglist[i].w.Length));
                        if (this.method == "adam" || this.method == "adadelta")
                        {
                            this.xsum.Add(util.zeros(pglist[i].w.Length));
                        }
                        else
                        {
                            this.xsum.Add(new List<double>().ToArray()); // conserve memory
                        }
                    }
                }

                // perform an update for all sets of weights
                for (var i = 0; i < pglist.Length; i++)
                {
                    var pg = pglist[i]; // param, gradient, other options in future (custom learning rate etc)
                    var p = pg.w;
                    var g = pg.dw;

                    // learning rate for some parameters.
                    var l2_decay_mul =pg.l2_decay_mul != double.MinValue ? pg.l2_decay_mul : 1.0;
                    var l1_decay_mul = pg.l1_decay_mul != double.MinValue ? pg.l1_decay_mul : 1.0;
                    var l2_decay = this.l2_decay * l2_decay_mul;
                    var l1_decay = this.l1_decay * l1_decay_mul;

                    var plen = p.Length;
                    for (var j = 0; j < plen; j++)
                    {
                        l2_decay_loss += l2_decay * p[j] * p[j] / 2; // accumulate weight decay loss
                        l1_decay_loss += l1_decay * Math.Abs(p[j]);
                        var l1grad = l1_decay * (p[j] > 0 ? 1 : -1);
                        var l2grad = l2_decay * (p[j]);

                        var gij = (l2grad + l1grad + g[j]) / this.batchSize; // raw batch gradient

                        var gsumi = this.gsum[i];
                        var xsumi = this.xsum[i];
                        if (this.method == "adam")
                        {
                            // adam update
                            gsumi[j] = gsumi[j] * this.beta1 + (1 - this.beta1) * gij; // update biased first moment estimate
                            xsumi[j] = xsumi[j] * this.beta2 + (1 - this.beta2) * gij * gij; // update biased second moment estimate
                            var biasCorr1 = gsumi[j] * (1 - Math.Pow(this.beta1, this.k)); // correct bias first moment estimate
                            var biasCorr2 = xsumi[j] * (1 - Math.Pow(this.beta2, this.k)); // correct bias second moment estimate
                            var dx = -this.learningRate * biasCorr1 / (Math.Sqrt(biasCorr2) + this.eps);
                            p[j] += dx;
                        }
                        else if (this.method == "adagrad")
                        {
                            // adagrad update
                            gsumi[j] = gsumi[j] + gij * gij;
                            var dx = -this.learningRate / Math.Sqrt(gsumi[j] + this.eps) * gij;
                            p[j] += dx;
                        }
                        else if (this.method == "windowgrad")
                        {
                            // this is adagrad but with a moving window weighted average
                            // so the gradient is not accumulated over the entire history of the run. 
                            // it's also referred to as Idea #1 in Zeiler paper on Adadelta. Seems reasonable to me!
                            gsumi[j] = this.ro * gsumi[j] + (1 - this.ro) * gij * gij;
                            var dx = -this.learningRate / Math.Sqrt(gsumi[j] + this.eps) * gij; // eps added for better conditioning
                            p[j] += dx;
                        }
                        else if (this.method == "adadelta")
                        {
                            gsumi[j] = this.ro * gsumi[j] + (1 - this.ro) * gij * gij;
                            var dx = -Math.Sqrt((xsumi[j] + this.eps) / (gsumi[j] + this.eps)) * gij;
                            xsumi[j] = this.ro * xsumi[j] + (1 - this.ro) * dx * dx; // yes, xsum lags behind gsum by 1.
                            p[j] += dx;
                        }
                        else if (this.method == "nesterov")
                        {
                            var dx = gsumi[j];
                            gsumi[j] = gsumi[j] * this.momentum + this.learningRate * gij;
                            dx = this.momentum * dx - (1.0 + this.momentum) * gsumi[j];
                            p[j] += dx;
                        }
                        else
                        {
                            // assume SGD
                            if (this.momentum > 0.0)
                            {
                                // momentum update
                                var dx = this.momentum * gsumi[j] - this.learningRate * gij; // step
                                gsumi[j] = dx; // back this up for next iteration of momentum
                                p[j] += dx; // apply corrected gradient
                            }
                            else
                            {
                                // vanilla sgd
                                p[j] += -this.learningRate * gij;
                            }
                        }
                        g[j] = 0.0; // zero out gradient so that we can begin accumulating anew
                    }
                }
            }

            // appending softmax_loss for backwards compatibility, but from now on we will always use cost_loss
            // in future, TODO: have to completely redo the way loss is done around the network as currently 
            // loss is a bit of a hack. Ideally, user should specify arbitrary number of loss functions on any layer
            // and it should all be computed correctly and automatically. 

            var result = new Dictionary<string, string>();
            result.Add("fwd_time", fwd_time.TotalMilliseconds + " millisec");
            result.Add("bwd_time", bwd_time.TotalMilliseconds + " millisec");
            result.Add("l2_decay_loss", l2_decay_loss.ToString());
            result.Add("l1_decay_loss", l1_decay_loss.ToString());
            result.Add("cost_loss", cost_loss.ToString());
            result.Add("loss", (cost_loss + l1_decay_loss + l2_decay_loss).ToString());

            return result;
        }
    }
}
