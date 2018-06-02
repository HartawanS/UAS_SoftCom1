/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package uas_softcom1;

/**
 *
 * @author wawan
 */
public class NNWeight {

    /**
     * @return the actual
     */
    public double[] getActual() {
        return actual;
    }

    /**
     * @return the persen
     */
    public double getPersen() {
        return persen;
    }

    private final int INPUT_NEURONS = 20;
    private final int HIDDEN_NEURONS = 10;
    private final int OUTPUT_NEURONS = 2;

    private final double LEARNING_RATE = 0.2;
    private int TRAINING_REPS;

    private final double momentum = 0.0001;
    
    // bobot pada hidden layer
    private double wih[][] = new double[INPUT_NEURONS + 1][HIDDEN_NEURONS];

    // bobot pada output layer
    private double who[][] = new double[HIDDEN_NEURONS + 1][OUTPUT_NEURONS];

    private double inputs[] = new double[INPUT_NEURONS];
    private double hidden[] = new double[HIDDEN_NEURONS];
    private double target[] = new double[OUTPUT_NEURONS];
    private double actual[] = new double[OUTPUT_NEURONS];

    // galat
    private double erro[] = new double[OUTPUT_NEURONS];
    private double errh[] = new double[HIDDEN_NEURONS];

    private final int MAX_SAMPLES = 613;

    private double trainInput[][] = new double[5000][INPUT_NEURONS];
    
    private int trainOutput[][] = new int[5000][OUTPUT_NEURONS];
    
    private double persen;
    
    public NNWeight(double inputs[], double WIH[][], double WHO[][], int epoch) {
//        inputs = new double[] {1,0,0,1,0,0,1,1,1,1,0,0,0.5,0,0,1,1};
         inputs = inputs;
        epoch = this.TRAINING_REPS;
        WHO = this.who;
        WIH = this.wih;
        feedForward();
        System.out.print("\n OUTPUT: ");

        for (int j = 0; j < OUTPUT_NEURONS; j++) {
          System.out.printf("%2.2f \t", actual[j]);
        }

        testNetworkTraining();
        getTrainingStats();
    }
    
    private void getTrainingStats()
    {
        double sum = 0.0;
        for(int i = 0; i < MAX_SAMPLES; i++)
        {
            for(int j = 0; j < INPUT_NEURONS; j++)
            {
                inputs[j] = trainInput[i][j];
            } // j

            for(int j = 0; j < OUTPUT_NEURONS; j++)
            {
                target[j] = trainOutput[i][j];
            } // j

            feedForward();

            if(maximum(getActual()) == maximum(target)){
                sum += 1;
            }else{
                //System.out.println(inputs[0] + "\t" + inputs[1] + "\t" + inputs[2] + "\t" + inputs[3]);
              System.out.println(maximum(getActual()) + "\t" + maximum(target));
            }
        } // i
        persen = ((double)sum / (double)MAX_SAMPLES * 100.0);
        System.out.println("Network is " + getPersen() + "% correct.");

        return;
    }
    
    private void feedForward() {
      double sum = 0;
      // hitung input ke hidden

      for (int hid = 0; hid < HIDDEN_NEURONS; hid++) {
        sum = 0.0;
        for (int inp = 0; inp < INPUT_NEURONS; inp++) {
          sum = sum + inputs[inp] * wih[inp][hid];
        }
        sum = sum + 1 * wih[INPUT_NEURONS][hid]; // bias
        hidden[hid] = sigmoid(sum);
      }

      // hitung hidden ke output
      for (int out = 0; out < OUTPUT_NEURONS; out++) {
        sum = 0.0;
        for (int hid = 0; hid < HIDDEN_NEURONS; hid++) {
          sum = sum + hidden[hid] * who[hid][out];
        }
        sum = sum + 1 * who[HIDDEN_NEURONS][out];
        actual[out] = sigmoid(sum);
      }
      return;

    }
    
    private double sigmoid(final double val) {
      return (1.0 / (1.0 + Math.exp(-val)));
    }
    
    private int maximum(final double[] vector)
    {
        // This function returns the index of the maximum of vector().
        int sel = 0;
        double max = vector[sel];

        for(int index = 0; index < OUTPUT_NEURONS; index++)
        {
            if(vector[index] > max){
                max = vector[index];
                sel = index;
            }
        }
        return sel;
    }
    
    private void testNetworkTraining() {
        // This function simply tests the training vectors against network.
        for(int i = 0; i < MAX_SAMPLES; i++) {
            for(int j = 0; j < INPUT_NEURONS; j++) {
                inputs[j] = trainInput[i][j];
            } // j

            feedForward();

            for(int j = 0; j < INPUT_NEURONS; j++) {
                //System.out.print(inputs[j] + "\t");
            } // j

            System.out.print("Output: " + (maximum(getActual())+1) + "\n");
        } // i

        return;
    }
    
    private void printWeights() {
      for (int inp = 0; inp <= INPUT_NEURONS; inp++) {
        for (int hid = 0; hid < HIDDEN_NEURONS; hid++) {
          System.out.printf("%2.2f \t",wih[inp][hid]);
        }
        System.out.println();
      }
    }

    /**
     * @return the TRAINING_REPS
     */
    public int getTRAINING_REPS() {
        return TRAINING_REPS;
    }
}
