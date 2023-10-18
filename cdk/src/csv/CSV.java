package csv;

import com.opencsv.CSVWriter;
import com.opencsv.bean.CsvToBeanBuilder;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class CSV {
    public List<Compound> read(String path, Class<? extends Compound> type) throws FileNotFoundException {
        return new CsvToBeanBuilder<Compound>(new FileReader(path)).withType(type).build().parse();
    }

    public void write(String path, List<Compound> compounds) throws IOException {
        CSVWriter writer = new CSVWriter(new FileWriter(path));

        List<String[]> data   = new ArrayList<>();
        List<String>   header = new ArrayList<>();
        header.add("SMILES");
        header.addAll(compounds.get(0).descriptors.keySet());
        data.add(header.toArray(new String[0]));

        compounds.forEach(compound -> {
            List<String> values = new ArrayList<>();
            values.add(compound.smiles);
            for (String name : header) {
                if (name.equals("SMILES")) continue;
                values.add(compound.descriptors.get(name));
            }
            data.add(values.toArray(new String[0]));
        });
        writer.writeAll(data);
        writer.close();
    }
}