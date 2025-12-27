import java.io.*;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;

public class CSVReader {

    public static List<MarketData> readCSV(String filename) {
        List<MarketData> dataList = new ArrayList<>();
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy.MM.dd HH:mm");

        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String line;
            br.readLine(); // ignore header

            while ((line = br.readLine()) != null) {
                String[] parts = line.split(";");
                LocalDateTime time = LocalDateTime.parse(parts[0], formatter);
                double open = Double.parseDouble(parts[1]);
                double high = Double.parseDouble(parts[2]);
                double low = Double.parseDouble(parts[3]);
                double close = Double.parseDouble(parts[4]);
                dataList.add(new MarketData(time, open, high, low, close));
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

        return dataList;
    }
}
