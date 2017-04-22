import de.uni_stuttgart.isa.liquidsvm.Config;
import de.uni_stuttgart.isa.liquidsvm.ResultAndErrors;
import de.uni_stuttgart.isa.liquidsvm.SVM;
import de.uni_stuttgart.isa.liquidsvm.SVM.LS;
import de.uni_stuttgart.isa.liquidsvm.LiquidData;

public class Example {

	public static void main(String[] args) throws java.io.IOException {
	
		String filePrefix = (args.length==0) ? "reg-1d" : args[0];
		
		// read comma separated training and testing data
		LiquidData data = new LiquidData(filePrefix);

		// Now train a least squares SVM on a 10by10 hyperparameter grid
		// and select the best parameters. The configuration displays
		// some progress information and specifies to only use two threads.
		SVM s = new LS(data.train, new Config().display(1).threads(2));

		// evaluate the selected SVM on the test features  
		double[] predictions = s.predict(data.testX);
		// or (since we have labels) do this and calculate the error
		ResultAndErrors result = s.test(data.test);
		
		System.out.println("Test error: " + result.errors[0][0]);
		for(int i=0; i<Math.min(result.result.length, 5); i++)
			System.out.println(predictions[i] + "==" + result.result[i][0]);

	}
}
