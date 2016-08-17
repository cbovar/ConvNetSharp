using System;
using System.Collections.Generic;
using System.Diagnostics;

namespace ConvNetSharp
{
    public class Trainer
    {
        public enum Method
        {
            Sgd, // Stochastic gradient descent
            Adam,
            Adagrad,
            Adadelta,
            Windowgrad,
            Netsterov
        }

        private readonly List<double[]> gsum = new List<double[]>(); // last iteration gradients (used for momentum calculations)
        private readonly Net net;
        private readonly List<double[]> xsum = new List<double[]>(); // used in adam or adadelta
        private int k; // iteration counter

        public Trainer(Net net)
        {
            this.net = net;

            this.LearningRate = 0.01;
            this.BatchSize = 1;
            this.TrainingMethod = Method.Sgd;
            this.Momentum = 0.9;
            this.Ro = 0.95;
            this.Eps = 1e-6;
            this.Beta1 = 0.9;
            this.Beta2 = 0.999;
        }

        public double L2DecayLoss { get; private set; }

        public double L1DecayLoss { get; private set; }

        public TimeSpan BackwardTime { get; private set; }

        public double CostLoss { get; private set; }

        public TimeSpan ForwardTime { get; private set; }

        public double LearningRate { get; set; }

        public double Ro { get; set; }  // used in adadelta

        public double Eps { get; set; } // used in adam or adadelta

        public double Beta1 { get; set; } // used in adam

        public double Beta2 { get; set; } // used in adam

        public double Momentum { get; set; }

        public double L1Decay { get; set; }

        public double L2Decay { get; set; }

        public int BatchSize { get; set; }

        public Method TrainingMethod { get; set; }

        public double Loss
        {
            get { return this.CostLoss + this.L1DecayLoss + this.L2DecayLoss; }
        }

        public void Train(Volume x, double y)
        {
            this.Forward(x);

            this.Backward(y);

            this.TrainImplem();
        }

        public void Train(Volume x, double[] y)
        {
            this.Forward(x);

            this.Backward(y);

            this.TrainImplem();
        }

        private void TrainImplem()
        {
            this.k++;
            if (this.k % this.BatchSize == 0)
            {
                List<ParametersAndGradients> parametersAndGradients = this.net.GetParametersAndGradients();

                // initialize lists for accumulators. Will only be done once on first iteration
                if (this.gsum.Count == 0 && (this.TrainingMethod != Method.Sgd || this.Momentum > 0.0))
                {
                    // only vanilla sgd doesnt need either lists
                    // momentum needs gsum
                    // adagrad needs gsum
                    // adam and adadelta needs gsum and xsum
                    for (var i = 0; i < parametersAndGradients.Count; i++)
                    {
                        this.gsum.Add(new double[parametersAndGradients[i].Parameters.Length]);
                        if (this.TrainingMethod == Method.Adam || this.TrainingMethod == Method.Adadelta)
                        {
                            this.xsum.Add(new double[parametersAndGradients[i].Parameters.Length]);
                        }
                    }
                }

                // perform an update for all sets of weights
                for (var i = 0; i < parametersAndGradients.Count; i++)
                {
                    var parametersAndGradient = parametersAndGradients[i];
                    // param, gradient, other options in future (custom learning rate etc)
                    double[] parameters = parametersAndGradient.Parameters;
                    double[] gradients = parametersAndGradient.Gradients;

                    // learning rate for some parameters.
                    var l2DecayMul = parametersAndGradient.L2DecayMul ?? 1.0;
                    var l1DecayMul = parametersAndGradient.L1DecayMul ?? 1.0;
                    var l2Decay = this.L2Decay * l2DecayMul;
                    var l1Decay = this.L1Decay * l1DecayMul;

                    var plen = parameters.Length;
                    for (var j = 0; j < plen; j++)
                    {
                        this.L2DecayLoss += l2Decay * parameters[j] * parameters[j] / 2; // accumulate weight decay loss
                        this.L1DecayLoss += l1Decay * Math.Abs(parameters[j]);
                        var l1Grad = l1Decay * (parameters[j] > 0 ? 1 : -1);
                        var l2Grad = l2Decay * (parameters[j]);

                        var gij = (l2Grad + l1Grad + gradients[j]) / this.BatchSize; // raw batch gradient

                        double[] gsumi = null;
                        if (this.gsum.Count > 0)
                        {
                            gsumi = this.gsum[i];
                        }

                        double[] xsumi = null;
                        if (this.xsum.Count > 0)
                        {
                            xsumi = this.xsum[i];
                        }

                        switch (this.TrainingMethod)
                        {
                            case Method.Sgd:
                                {
                                    if (this.Momentum > 0.0)
                                    {
                                        // momentum update
                                        var dx = this.Momentum * gsumi[j] - this.LearningRate * gij; // step
                                        gsumi[j] = dx; // back this up for next iteration of momentum
                                        parameters[j] += dx; // apply corrected gradient
                                    }
                                    else
                                    {
                                        // vanilla sgd
                                        parameters[j] += -this.LearningRate * gij;
                                    }
                                }
                                break;
                            case Method.Adam:
                                {
                                    // adam update
                                    gsumi[j] = gsumi[j] * this.Beta1 + (1 - this.Beta1) * gij; // update biased first moment estimate
                                    xsumi[j] = xsumi[j] * this.Beta2 + (1 - this.Beta2) * gij * gij; // update biased second moment estimate
                                    var biasCorr1 = gsumi[j] * (1 - Math.Pow(this.Beta1, this.k)); // correct bias first moment estimate
                                    var biasCorr2 = xsumi[j] * (1 - Math.Pow(this.Beta2, this.k)); // correct bias second moment estimate
                                    var dx = -this.LearningRate * biasCorr1 / (Math.Sqrt(biasCorr2) + this.Eps);
                                    parameters[j] += dx;
                                }
                                break;
                            case Method.Adagrad:
                                {
                                    // adagrad update
                                    gsumi[j] = gsumi[j] + gij * gij;
                                    var dx = -this.LearningRate / Math.Sqrt(gsumi[j] + this.Eps) * gij;
                                    parameters[j] += dx;
                                }
                                break;
                            case Method.Adadelta:
                                {
                                    // assume adadelta if not sgd or adagrad
                                    gsumi[j] = this.Ro * gsumi[j] + (1 - this.Ro) * gij * gij;
                                    var dx = -Math.Sqrt((xsumi[j] + this.Eps) / (gsumi[j] + this.Eps)) * gij;
                                    xsumi[j] = this.Ro * xsumi[j] + (1 - this.Ro) * dx * dx; // yes, xsum lags behind gsum by 1.
                                    parameters[j] += dx;
                                }
                                break;
                            case Method.Windowgrad:
                                {
                                    // this is adagrad but with a moving window weighted average
                                    // so the gradient is not accumulated over the entire history of the run. 
                                    // it's also referred to as Idea #1 in Zeiler paper on Adadelta. Seems reasonable to me!
                                    gsumi[j] = this.Ro * gsumi[j] + (1 - this.Ro) * gij * gij;
                                    var dx = -this.LearningRate / Math.Sqrt(gsumi[j] + this.Eps) * gij;
                                    // eps added for better conditioning
                                    parameters[j] += dx;
                                }
                                break;
                            case Method.Netsterov:
                                {
                                    var dx = gsumi[j];
                                    gsumi[j] = gsumi[j] * this.Momentum + this.LearningRate * gij;
                                    dx = this.Momentum * dx - (1.0 + this.Momentum) * gsumi[j];
                                    parameters[j] += dx;
                                }
                                break;
                            default:
                                throw new ArgumentOutOfRangeException();
                        }

                        gradients[j] = 0.0; // zero out gradient so that we can begin accumulating anew
                    }
                }
            }

            // appending softmax_loss for backwards compatibility, but from now on we will always use cost_loss
            // in future, TODO: have to completely redo the way loss is done around the network as currently 
            // loss is a bit of a hack. Ideally, user should specify arbitrary number of loss functions on any layer
            // and it should all be computed correctly and automatically. 
        }

        private void Backward(double y)
        {
            var chrono = Stopwatch.StartNew();
            this.CostLoss = this.net.Backward(y);
            this.L2DecayLoss = 0.0;
            this.L1DecayLoss = 0.0;
            this.BackwardTime = chrono.Elapsed;
        }

        private void Backward(double[] y)
        {
            var chrono = Stopwatch.StartNew();
            this.CostLoss = this.net.Backward(y);
            this.L2DecayLoss = 0.0;
            this.L1DecayLoss = 0.0;
            this.BackwardTime = chrono.Elapsed;
        }

        private void Forward(Volume x)
        {
            var chrono = Stopwatch.StartNew();
            this.net.Forward(x, true); // also set the flag that lets the net know we're just training
            this.ForwardTime = chrono.Elapsed;
        }
    }
}