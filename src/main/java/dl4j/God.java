package dl4j;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;

public class God {
    private static final long seed = 1337;
    private int epochs = 300;
    private int outputNum = 3;

    public static void main(String[] args) throws Exception {
        new God().start(args);
    }

    private void start(String[] args) throws Exception {
        final MultiLayerNetwork model = buildModel(1, 0.1, 100, 20, outputNum, 10);
        attachStatsServer(model);
        DataSetIterator dataIter = createDataSetIterator();
        train(model, dataIter);
        evaluate(model, dataIter);
    }

    /**
     * Output Interpretation https://deeplearning4j.org/output
     */
    private void evaluate(MultiLayerNetwork model, DataSetIterator dataIter) {
        Evaluation eval = new Evaluation(outputNum);

        while (dataIter.hasNext()) {
            org.nd4j.linalg.dataset.api.DataSet t = dataIter.next();
            INDArray features = t.getFeatureMatrix();
            INDArray labels = t.getLabels();
            INDArray predicted = model.output(features, false);

            eval.eval(labels, predicted);
        }
        System.out.println(eval.stats());
    }

    /**
     * Early stopping https://deeplearning4j.org/earlystopping
     */
    private void train(MultiLayerNetwork model, DataSetIterator dataIter) {
        for (int n = 0; n < epochs; n++) {
            model.fit(dataIter);
        }
    }

    /**
     * Restricted Boltzmann machine https://deeplearning4j.org/restrictedboltzmannmachine
     *
     * Invented by Geoff Hinton, a Restricted Boltzmann machine is an algorithm useful for dimensionality reduction,
     * classification, regression, collaborative filtering, feature learning and topic modeling. (For more concrete
     * examples of how neural networks like RBMs can be employed, please see our page on use cases).
     *
     * Given their relative simplicity and historical importance, restricted Boltzmann machines are the first neural
     * network we’ll tackle. In the paragraphs below, we describe in diagrams and plain language how they work.
     */
    private MultiLayerNetwork buildModel(int iterations, double learningRate, int inSize, int numHiddenNodes, int outputNum, int feedForwardSize) {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(learningRate)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(inSize).nOut(numHiddenNodes)
                        .weightInit(WeightInit.XAVIER)
                        .activation("relu")
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .weightInit(WeightInit.XAVIER)
                        .activation("softmax").weightInit(WeightInit.XAVIER)
                        .nIn(numHiddenNodes).nOut(outputNum).build())
                .pretrain(false).backprop(true)
                .setInputType(InputType.feedForward(feedForwardSize))
                .build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        return model;
    }

    /**
     * Visualization https://deeplearning4j.org/visualization
     */
    private void attachStatsServer(MultiLayerNetwork model) throws IOException {
        File ui = File.createTempFile("ui-server-dl4j", "bin");
        ui.delete();
        StatsStorage statsStorage = new FileStatsStorage(ui);             //Alternative: new FileStatsStorage(File) - see UIStorageExample
        model.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(1));

        UIServer uiServer = UIServer.getInstance();
        uiServer.attach(statsStorage);
    }

    /**
     * Look here https://deeplearning4j.org/datavec
     */
    private DataSetIterator createDataSetIterator() {
        throw new UnsupportedOperationException("TODO");
    }
}
