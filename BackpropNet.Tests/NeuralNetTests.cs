namespace BackpropNet.Tests;

public class NeuralNet_MeanSquaredError_Tests
{
    [Fact]
    public void Test_CanComputeLoss()
    {
        var lossFunc = new MeanSquaredError();
        var pred = Matrix2D.FromData(2, 3, new double[] { 0, 1, 2, 3, 4, 5 });
        var truthSame = Matrix2D.FromData(2, 3, new double[] { 0, 1, 2, 3, 4, 5 });
        var truth1Dist = Matrix2D.FromData(2, 3, new double[] { 1, 2, 3, 4, 5, 6 });

        double lossSameData = lossFunc.Loss(pred, truthSame);
        double lossDistData = lossFunc.Loss(pred, truth1Dist);

        lossSameData.Should().BeApproximately(0, 1e-5);
        lossDistData.Should().BeApproximately(3, 1e-5);
    }

    [Fact]
    public void Test_CanComputeDeltas()
    {
        var lossFunc = new MeanSquaredError();
        var pred = Matrix2D.FromData(2, 3, new double[] { 0, 1, 2, 3, 4, 5 });
        var truth = Matrix2D.FromData(2, 3, new double[] { 1, 2, 3, 4, 5, 6 });

        var deltas = Matrix2D.Zeros(2, 3);
        lossFunc.LossDeltas(pred, truth, deltas);

        var exp = Matrix2D.FromData(2, 3, new double[] { -0.5, -0.5, -0.5, -0.5, -0.5, -0.5 });
        deltas.Should().BeEquivalentTo(exp);
    }
}

public class NeuralNet_CrossEntropyLoss_Tests
{
    [Fact]
    public void Test_CanComputeLoss()
    {
        var lossFunc = new CrossEntropyLoss();
        var pred = Matrix2D.FromData(2, 3, new double[] { 0, 1, 0, 1, 0, 0 });
        var truthSame = Matrix2D.FromData(2, 3, new double[] { 0, 1, 0, 1, 0, 0 });
        var truthDist = Matrix2D.FromData(2, 3, new double[] { 1, 0, 0, 0, 1, 0 });

        double lossSameData = lossFunc.Loss(pred, truthSame);
        double lossDistData = lossFunc.Loss(pred, truthDist);

        lossSameData.Should().BeApproximately(0, 1e-4);
        Math.Abs(lossDistData).Should().BeGreaterThan(1);
    }

    [Fact]
    public void Test_CanComputeDeltas()
    {
        var lossFunc = new CrossEntropyLoss();
        var pred = Matrix2D.FromData(2, 3, new double[] { 0, 1, 0, 0, 0, 1 });
        var truthSame = Matrix2D.FromData(2, 3, new double[] { 0, 1, 0, 0, 0, 1 });
        var truthDist = Matrix2D.FromData(2, 3, new double[] { 1, 0, 0, 0, 0, 1 });

        var deltasSame = Matrix2D.Zeros(2, 3);
        var deltasDist = Matrix2D.Zeros(2, 3);
        lossFunc.LossDeltas(pred, truthSame, deltasSame);
        lossFunc.LossDeltas(pred, truthDist, deltasDist);

        deltasSame.Should().BeEquivalentTo(truthSame);
        deltasDist.Should().BeEquivalentTo(truthDist);
    }
}

public class NeuralNet_SparseCrossEntropyLoss_Tests
{
    [Fact]
    public void Test_CanComputeLoss()
    {
        var lossFunc = new SparseCrossEntropyLoss();
        var pred = Matrix2D.FromData(2, 3, new double[] { 0, 1, 0, 1, 0, 0 });
        var truthSame = Matrix2D.FromData(2, 1, new double[] { 1, 0 });
        var truthDist = Matrix2D.FromData(2, 1, new double[] { 0, 1 });

        double lossSameData = lossFunc.Loss(pred, truthSame);
        double lossDistData = lossFunc.Loss(pred, truthDist);

        lossSameData.Should().BeApproximately(0, 1e-4);
        Math.Abs(lossDistData).Should().BeGreaterThan(1);
    }

    [Fact]
    public void Test_CanComputeDeltas()
    {
        var lossFunc = new SparseCrossEntropyLoss();
        var pred = Matrix2D.FromData(2, 3, new double[] { 0, 1, 0, 1, 0, 0 });
        var truthSame = Matrix2D.FromData(2, 1, new double[] { 1, 0 });
        var truthDist = Matrix2D.FromData(2, 1, new double[] { 2, 1 });

        var deltasSame = Matrix2D.Zeros(2, 3);
        var deltasDist = Matrix2D.Zeros(2, 3);
        lossFunc.LossDeltas(pred, truthSame, deltasSame);
        lossFunc.LossDeltas(pred, truthDist, deltasDist);

        var expDeltasSame = Matrix2D.FromData(2, 3, new double[] { 0, 1, 0, 1, 0, 0 });
        var expDeltasDist = Matrix2D.FromData(2, 3, new double[] { 0, 0, 1, 0, 1, 0 });
        deltasSame.Should().BeEquivalentTo(expDeltasSame);
        deltasDist.Should().BeEquivalentTo(expDeltasDist);
    }
}

public class NeuralNet_DenseLayer_Tests
{
    [Fact]
    public void Test_CanInitWeightsAndBiases()
    {
        const int batchSize = 2;
        const int inputDims = 3;
        const int outputDims = 1;
        var input = Matrix2D.FromData(batchSize, inputDims, new double[] { 0, 1, 2, 3, 4, 5 });
        var deltasOut = Matrix2D.Zeros(batchSize, inputDims);

        var dense = new DenseLayer(outputDims);
        dense.Compile(inputDims);
        dense.CompileCache(input, deltasOut);

        dense.Weights.NumRows.Should().Be(inputDims);
        dense.Weights.NumCols.Should().Be(outputDims);
        dense.Biases.NumRows.Should().Be(1);
        dense.Biases.NumCols.Should().Be(outputDims);
    }

    [Fact]
    public void Test_CanInitLayerCache()
    {
        const int batchSize = 2;
        const int inputDims = 3;
        const int outputDims = 1;
        var input = Matrix2D.FromData(batchSize, inputDims, new double[] { 0, 1, 2, 3, 4, 5 });
        var deltasOut = Matrix2D.Zeros(batchSize, inputDims);

        var dense = new DenseLayer(outputDims);
        dense.Compile(inputDims);
        dense.CompileCache(input, deltasOut);

        dense.Cache.Input.NumRows.Should().Be(batchSize);
        dense.Cache.Input.NumCols.Should().Be(inputDims);
        dense.Cache.Output.NumRows.Should().Be(batchSize);
        dense.Cache.Output.NumCols.Should().Be(outputDims);
        dense.Cache.DeltasIn.NumRows.Should().Be(batchSize);
        dense.Cache.DeltasIn.NumCols.Should().Be(outputDims);
        dense.Cache.DeltasOut.NumRows.Should().Be(batchSize);
        dense.Cache.DeltasOut.NumCols.Should().Be(inputDims);
        dense.Cache.Gradients.NumRows.Should().Be(1);
        dense.Cache.Gradients.NumCols.Should().Be((inputDims + 1) * outputDims);
    }

    [Fact]
    public void Test_CanProcessForwardPass()
    {
        var input = Matrix2D.FromData(2, 3, new double[] { 0, 1, 2, 3, 4, 5 });
        var weights = Matrix2D.FromData(3, 1, new double[] { 0, 1, 2 });
        var biases = Matrix2D.FromData(1, 1, new double[] { 1 });
        var deltasOut = Matrix2D.Zeros(2, 3);
        var dense = new DenseLayer(1);
        dense.Compile(3);
        dense.CompileCache(input, deltasOut);
        dense.Load(new Matrix2D[] { weights, biases });

        dense.Forward();
        var pred = dense.Cache.Output;

        var exp = Matrix2D.FromData(2, 1, new double[] { 5 + 1, 14 + 1 });
        pred.Should().BeEquivalentTo(exp);
    }

    [Fact]
    public void Test_CanProcessBackwardPass()
    {
        var input = Matrix2D.FromData(2, 3, new double[] { 0, 1, 2, 3, 4, 5 });
        var yTrue = Matrix2D.FromData(2, 1, new double[] { 7, 14 });
        var weights = Matrix2D.FromData(3, 1, new double[] { 0, 1, 2 });
        var biases = Matrix2D.FromData(1, 1, new double[] { 1 });
        var deltasOut = Matrix2D.Zeros(2, 3);
        var dense = new DenseLayer(1);
        dense.Compile(3);
        dense.CompileCache(input, deltasOut);
        dense.Load(new Matrix2D[] { weights, biases });
        var loss = new MeanSquaredError();

        dense.Forward();
        loss.LossDeltas(dense.Cache.Output, yTrue, dense.Cache.DeltasIn);
        dense.Backward();

        // input^T: [[0, 3],
        //           [1, 4],
        //           [2, 5]]

        // deltas:  [[-0.5],
        //           [ 0.5]]

        var expWeightGrads = Matrix2D.FromData(3, 1, new double[] { 1.5, 1.5, 1.5 });
        var expBiasGrads = Matrix2D.FromData(1, 1, new double[] { 0 });
        var expDeltasOut = Matrix2D.FromData(2, 3, new double[] { 0, -0.5, -1, 0, 0.5, 1 });
        dense.WeightGrads.Should().BeEquivalentTo(expWeightGrads);
        dense.BiasGrads.Should().BeEquivalentTo(expBiasGrads);
        dense.Cache.DeltasOut.Should().BeEquivalentTo(expDeltasOut);
    }

    [Fact]
    public void Test_CanOptimizePrediction()
    {
        var input = Matrix2D.FromData(2, 3, new double[] { 0, 1, 2, 3, 4, 5 });
        var yTrue = Matrix2D.FromData(2, 1, new double[] { 7, 14 });
        var weights = Matrix2D.FromData(3, 1, new double[] { 0, 1, 2 });
        var biases = Matrix2D.FromData(1, 1, new double[] { 1 });
        var deltasOut = Matrix2D.Zeros(2, 3);
        var dense = new DenseLayer(1);
        dense.Compile(3);
        dense.CompileCache(input, deltasOut);
        dense.Load(new Matrix2D[] { weights, biases });
        var lossFunc = new MeanSquaredError();
        var opt = new AdamOpt(0.01);
        var modelGrads = new Matrix2D[] { dense.Cache.Gradients };
        opt.Compile(modelGrads);

        for (int i = 0; i < 1000; i++)
        {
            dense.Forward();
            lossFunc.LossDeltas(dense.Cache.Output, yTrue, dense.Cache.DeltasIn);
            dense.Backward();
            opt.AdjustGrads(modelGrads);
            dense.ApplyGrads();
        }

        double loss = lossFunc.Loss(dense.Cache.Output, yTrue);
        loss.Should().BeLessThan(0.01);
    }
}

public class NeuralNet_ReLULayer_Tests
{
    [Fact]
    public void Test_CanInitLayerCache()
    {
        const int batchSize = 2; const int dims = 3;
        var input = Matrix2D.FromData(batchSize, dims, new double[] { 0, 1, 2, 3, 4, 5 });
        var deltasOut = Matrix2D.Zeros(batchSize, dims);

        var relu = new ReLULayer();
        relu.Compile(3);
        relu.CompileCache(input, deltasOut);

        relu.Cache.Input.NumRows.Should().Be(batchSize);
        relu.Cache.Input.NumCols.Should().Be(dims);
        relu.Cache.Output.NumRows.Should().Be(batchSize);
        relu.Cache.Output.NumCols.Should().Be(dims);
        relu.Cache.DeltasIn.NumRows.Should().Be(batchSize);
        relu.Cache.DeltasIn.NumCols.Should().Be(dims);
        relu.Cache.DeltasOut.NumRows.Should().Be(batchSize);
        relu.Cache.DeltasOut.NumCols.Should().Be(dims);
        relu.Cache.Gradients.Should().BeEquivalentTo(Matrix2D.Null());
    }

    [Fact]
    public void Test_CanProcessForwardPass()
    {
        var input = Matrix2D.FromData(2, 3, new double[] { -2, -1, 0, 1, 2, 3 });
        var deltasOut = Matrix2D.Zeros(2, 3);
        var relu = new ReLULayer();
        relu.Compile(3);
        relu.CompileCache(input, deltasOut);

        relu.Forward();

        var expOutput = Matrix2D.FromData(2, 3, new double[] { 0, 0, 0, 1, 2, 3 });
        relu.Cache.Output.Should().BeEquivalentTo(expOutput);
    }

    [Fact]
    public void Test_CanProcessBackwardPass()
    {
        var input = Matrix2D.FromData(2, 3, new double[] { -2, -1, 0, 1, 2, 3 });
        var deltasIn = Matrix2D.FromData(2, 3, new double[] { 1, 1, 1, 1, 1, 1 });
        var deltasOut = Matrix2D.Zeros(2, 3);
        var relu = new ReLULayer();
        relu.Compile(3);
        relu.CompileCache(input, deltasOut);

        relu.Forward();
        relu.Cache.DeltasIn = deltasIn;
        relu.Backward();

        var expDeltasOut = Matrix2D.FromData(2, 3, new double[] { 0, 0, 1, 1, 1, 1 });
        relu.Cache.DeltasOut.Should().BeEquivalentTo(expDeltasOut);
    }
}

public class NeuralNet_SoftmaxLayer_Tests
{
    [Fact]
    public void Test_CanInitLayerCache()
    {
        const int batchSize = 2; const int dims = 3;
        var input = Matrix2D.FromData(batchSize, dims, new double[] { 0, 1, 2, 3, 4, 5 });
        var deltasOut = Matrix2D.Zeros(batchSize, dims);

        var softmax = new SoftmaxLayer();
        softmax.Compile(3);
        softmax.CompileCache(input, deltasOut);

        softmax.Cache.Input.NumRows.Should().Be(batchSize);
        softmax.Cache.Input.NumCols.Should().Be(dims);
        softmax.Cache.Output.NumRows.Should().Be(batchSize);
        softmax.Cache.Output.NumCols.Should().Be(dims);
        softmax.Cache.DeltasIn.NumRows.Should().Be(batchSize);
        softmax.Cache.DeltasIn.NumCols.Should().Be(dims);
        softmax.Cache.DeltasOut.NumRows.Should().Be(batchSize);
        softmax.Cache.DeltasOut.NumCols.Should().Be(dims);
        softmax.Cache.Gradients.Should().BeEquivalentTo(Matrix2D.Null());
    }

    [Fact]
    public void Test_CanProcessForwardPass()
    {
        var input = Matrix2D.FromData(2, 3, new double[] { -2, -1, 0, 1, 2, 3 });
        var deltasOut = Matrix2D.Zeros(2, 3);
        var softmax = new SoftmaxLayer();
        softmax.Compile(3);
        softmax.CompileCache(input, deltasOut);

        softmax.Forward();

        var sums = Matrix2D.Zeros(2, 1);
        Matrix2D.RowSum(softmax.Cache.Output, sums);
        Math.Abs(sums.At(0, 0)).Should().BeApproximately(1, 1e-4);
        Math.Abs(sums.At(1, 0)).Should().BeApproximately(1, 1e-4);
    }

    [Fact]
    public void Test_CanProcessBackwardPass()
    {
        var input = Matrix2D.FromData(2, 3, new double[] { 0, 1, 0, 1, 0, 0 });
        var output = Matrix2D.FromData(2, 3, new double[] { 0, 1, 0, 1, 0, 0 });
        var deltasIn = Matrix2D.FromData(2, 3, new double[] { 1, 0, 0, 1, 0, 0 });
        var deltasOut = Matrix2D.Zeros(2, 3);
        var softmax = new SoftmaxLayer();
        softmax.Compile(3);
        softmax.CompileCache(input, deltasOut);

        softmax.Cache.DeltasIn = deltasIn;
        softmax.Cache.Output = output;
        softmax.Backward();

        var expDeltasOut = Matrix2D.FromData(2, 3, new double[] { -1, 1, 0, 0, 0, 0 });
        softmax.Cache.DeltasOut.Should().BeEquivalentTo(expDeltasOut);
    }
}

public class NeuralNet_FeedForwardModel_Tests
{
    [Fact]
    public void Test_CanCompileNet()
    {
        const int batchSize = 32; const int inputDims = 10; const int outputDims = 1;
        var layers = new ILayer[] {
            new DenseLayer(64),
            new ReLULayer(),
            new DenseLayer(outputDims)
        };
        var model = new FFModel(layers);

        model.Compile(batchSize, inputDims);

        layers[0].Cache.Input.NumRows.Should().Be(batchSize);
        layers[0].Cache.Input.NumCols.Should().Be(inputDims);
        layers[2].Cache.Output.NumRows.Should().Be(batchSize);
        layers[2].Cache.Output.NumCols.Should().Be(outputDims);
    }

    [Fact]
    public void Test_CanComputeForwardPass()
    {
        var inputs = Matrix2D.RandNorm(4, 4, 0, 0.1);
        var weights = Matrix2D.FromData(4, 4, new double[] {
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1
        });
        var biases = Matrix2D.Zeros(1, 4);
        var dense1 = new DenseLayer(4);
        var dense2 = new DenseLayer(4);
        var model = new FFModel(new ILayer[] { dense1, dense2 });
        model.Compile(4, 4);
        dense1.Load(new Matrix2D[] { weights, biases });
        dense2.Load(new Matrix2D[] { weights, biases });

        var pred = model.PredictBatch(inputs);

        pred.Should().BeEquivalentTo(inputs);
    }
}
