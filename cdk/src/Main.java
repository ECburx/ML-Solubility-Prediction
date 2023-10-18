import csv.Compound;

import java.io.IOException;
import java.util.List;

import csv.CSV;
import org.openscience.cdk.exception.CDKException;

/**
 * SMILES Featurization using Java CDK library.
 */
public class Main {
    public static final String OUTPUT_PATH = "output.csv";

    public static void main(String[] args) throws IOException, CDKException {
        assert args.length == 1;
        String csvInputPath = args[0];
        System.out.println("Featurizing " + csvInputPath);

        CSV            csv       = new CSV();
        List<Compound> compounds = csv.read(csvInputPath, Compound.class);
        compounds = new Featurizer().featurize(compounds);
        csv.write(OUTPUT_PATH, compounds);
    }
}