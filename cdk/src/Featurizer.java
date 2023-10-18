import csv.Compound;
import org.openscience.cdk.DefaultChemObjectBuilder;
import org.openscience.cdk.exception.CDKException;
import org.openscience.cdk.exception.InvalidSmilesException;
import org.openscience.cdk.qsar.DescriptorValue;
import org.openscience.cdk.qsar.IMolecularDescriptor;
import org.openscience.cdk.qsar.descriptors.molecular.*;
import org.openscience.cdk.smiles.SmilesParser;

import java.util.List;

public class Featurizer {
    public List<IMolecularDescriptor> descriptors;

    public static final SmilesParser parser = new SmilesParser(
            DefaultChemObjectBuilder.getInstance()
    );

    public Featurizer() throws CDKException {
        descriptors = List.of(
                new ALOGPDescriptor(),
                new AminoAcidCountDescriptor(),
                new APolDescriptor(),
                new AromaticAtomsCountDescriptor(),
                new AromaticBondsCountDescriptor(),
                new AtomCountDescriptor(),
                new AutocorrelationDescriptorCharge(),
                new AutocorrelationDescriptorMass(),
                new AutocorrelationDescriptorPolarizability(),
                new BCUTDescriptor(),
                new BondCountDescriptor(),
                new BPolDescriptor(),
                new CarbonTypesDescriptor(),
                new ChiChainDescriptor(),
                new ChiClusterDescriptor(),
                new ChiPathClusterDescriptor(),
                new ChiPathDescriptor(),
                new EccentricConnectivityIndexDescriptor(),
                new FMFDescriptor(),
                new FractionalCSP3Descriptor(),
                new FractionalPSADescriptor(),
                new FragmentComplexityDescriptor(),
                new HBondAcceptorCountDescriptor(),
                new HBondDonorCountDescriptor(),
                new HybridizationRatioDescriptor(),
                new JPlogPDescriptor(),
                new KappaShapeIndicesDescriptor(),
                new KierHallSmartsDescriptor(),
                new LargestChainDescriptor(),
                new LargestPiSystemDescriptor(),
                new MannholdLogPDescriptor(),
                new MDEDescriptor(),
                new PetitjeanNumberDescriptor(),
                new RuleOfFiveDescriptor(),
                new SmallRingDescriptor(),
                new SpiroAtomCountDescriptor(),
                new TPSADescriptor(),
                new VAdjMaDescriptor(),
                new WeightDescriptor(),
                new WeightedPathDescriptor(),
                new WienerNumbersDescriptor(),
                new XLogPDescriptor(),
                new ZagrebIndexDescriptor()
        );
    }

    public List<Compound> featurize(List<Compound> compounds) {
        for (Compound compound : compounds) {
            compound = fromSMILES(compound);
            if (compound == null) continue;
            for (IMolecularDescriptor descriptor : descriptors) {
                try {
                    DescriptorValue description = descriptor.calculate(compound.molecule);
                    String[]        values      = description.getValue().toString().split(",");
                    String[]        names       = description.getNames();
                    for (int i = 0; i < names.length; i++) {
                        compound.descriptors.put(names[i], values[i]);
                    }
                } catch (Exception e) {
                    System.err.println(compound.smiles + ": " + e.getMessage());
                }
            }
        }
        return compounds;
    }

    public static Compound fromSMILES(Compound compound) {
        try {
            compound.molecule = parser.parseSmiles(compound.smiles);
            return compound;
        } catch (InvalidSmilesException e) {
            System.err.printf("[%s]: %s%n\n", compound.smiles, e);
            return null;
        }
    }
}
