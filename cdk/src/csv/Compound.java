package csv;

import org.openscience.cdk.interfaces.IAtomContainer;

import java.util.LinkedHashMap;
import java.util.Map;

public class Compound {
    public IAtomContainer molecule;
    public String         smiles;

    public Map<String, String> descriptors = new LinkedHashMap<>();
}
