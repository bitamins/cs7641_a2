/**
 * from reference here: https://github.com/JonathanTay/CS-7641-assignment-2/blob/master/ABAGAIL/src/func/nn/activation/RELU.java
 */


package func.nn.activation;

public class RELU extends DifferentiableActivationFunction{

	public double derivative(double value) {
		if (value < 0){
			return 0;
		} else {
			return 1;
		}
    }
    
    public double value(double value) {
        
        if (value < 0) {
            return 0;
        } else {
            return value;
        }
	}


}