import javafx.application.Application;
import javafx.geometry.Insets;
import javafx.scene.Scene;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.ScatterChart;
import javafx.scene.control.*;
import javafx.scene.control.cell.PropertyValueFactory;
import javafx.scene.layout.*;
import javafx.stage.FileChooser;
import javafx.stage.Stage;

import java.io.File;
import java.io.FileWriter;
import java.util.List;
import java.util.stream.Collectors;

public class Main extends Application {

    @Override
    public void start(Stage primaryStage) {
        List<MarketData> data = CSVReader.readCSV("MarketData_OHLC.csv");
        TrendAnalyzer analyzer = new TrendAnalyzer(100, 20);
        List<Trend> trends = analyzer.analyze(data);

        // ======= TableView =======
        TableView<Trend> table = new TableView<>();
        table.setColumnResizePolicy(TableView.CONSTRAINED_RESIZE_POLICY);
        table.setPlaceholder(new Label("Aucune tendance trouvée"));

        TableColumn<Trend, String> typeCol = new TableColumn<>("Type");
        typeCol.setCellValueFactory(new PropertyValueFactory<>("type"));

        TableColumn<Trend, Integer> durCol = new TableColumn<>("Duration");
        durCol.setCellValueFactory(new PropertyValueFactory<>("durationMinutes"));

        TableColumn<Trend, Double> moveCol = new TableColumn<>("Move Pips");
        moveCol.setCellValueFactory(new PropertyValueFactory<>("movePips"));

        TableColumn<Trend, Double> pullCol = new TableColumn<>("Pullback Pips");
        pullCol.setCellValueFactory(new PropertyValueFactory<>("pullbackPips"));

        table.getColumns().addAll(typeCol, durCol, moveCol, pullCol);
        table.getItems().addAll(trends);

        // Style alterné des lignes
        table.setRowFactory(tv -> new TableRow<>() {
            @Override
            protected void updateItem(Trend item, boolean empty) {
                super.updateItem(item, empty);
                if (item == null || empty) {
                    setStyle("");
                } else if (getIndex() % 2 == 0) {
                    setStyle("-fx-background-color: #f0f0f0;");
                } else {
                    setStyle("");
                }
            }
        });

        // ======= ScatterChart =======
        NumberAxis xAxis = new NumberAxis();
        xAxis.setLabel("Duration Minutes");
        NumberAxis yAxis = new NumberAxis();
        yAxis.setLabel("Move Pips");

        ScatterChart<Number, Number> chart = new ScatterChart<>(xAxis, yAxis);
        chart.setTitle("Trend Move vs Duration");

        XYChart.Series<Number, Number> bullishSeries = new XYChart.Series<>();
        bullishSeries.setName("BULLISH");
        XYChart.Series<Number, Number> bearishSeries = new XYChart.Series<>();
        bearishSeries.setName("BEARISH");

        for (Trend t : trends) {
            if ("BULLISH".equals(t.type))
                bullishSeries.getData().add(new XYChart.Data<>(t.durationMinutes, t.movePips));
            else
                bearishSeries.getData().add(new XYChart.Data<>(t.durationMinutes, t.movePips));
        }

        chart.getData().addAll(bullishSeries, bearishSeries);

        // ======= Statistiques Globales =======
        int bullishCount = (int) trends.stream().filter(t -> t.type.equals("BULLISH")).count();
        int bearishCount = (int) trends.stream().filter(t -> t.type.equals("BEARISH")).count();
        double avgMove = trends.stream().collect(Collectors.averagingDouble(t -> t.movePips));
        double avgPull = trends.stream().collect(Collectors.averagingDouble(t -> t.pullbackPips));

        Label statsLabel = new Label(
                "Stats: BULLISH=" + bullishCount +
                        " | BEARISH=" + bearishCount +
                        " | Avg Move=" + String.format("%.2f", avgMove) +
                        " | Avg Pullback=" + String.format("%.2f", avgPull)
        );
        statsLabel.setPadding(new Insets(5));

        // ======= SplitPane pour Table + Chart =======
        SplitPane splitPane = new SplitPane();
        VBox tableBox = new VBox(table, statsLabel);
        tableBox.setSpacing(5);
        tableBox.setPadding(new Insets(5));
        splitPane.getItems().addAll(chart, tableBox);
        splitPane.setDividerPositions(0.6);

        // ======= Menu =======
        MenuBar menuBar = new MenuBar();
        Menu fileMenu = new Menu("File");
        MenuItem exportItem = new MenuItem("Export CSV");
        MenuItem exitItem = new MenuItem("Exit");
        fileMenu.getItems().addAll(exportItem, exitItem);
        menuBar.getMenus().add(fileMenu);

        exportItem.setOnAction(e -> exportCSV(trends, primaryStage));
        exitItem.setOnAction(e -> primaryStage.close());

        VBox root = new VBox(menuBar, splitPane);

        Scene scene = new Scene(root, 1000, 650);
        primaryStage.setTitle("Professional Trend Analyzer");
        primaryStage.setScene(scene);
        primaryStage.show();
    }

    private void exportCSV(List<Trend> trends, Stage stage) {
        FileChooser fileChooser = new FileChooser();
        fileChooser.setTitle("Save CSV");
        fileChooser.getExtensionFilters().add(new FileChooser.ExtensionFilter("CSV Files", "*.csv"));
        File file = fileChooser.showSaveDialog(stage);
        if (file != null) {
            try (FileWriter fw = new FileWriter(file)) {
                fw.write("Type,Duration,MovePips,PullbackPips\n");
                for (Trend t : trends) {
                    fw.write(String.format("%s,%d,%.2f,%.2f\n",
                            t.type, t.durationMinutes, t.movePips, t.pullbackPips));
                }
            } catch (Exception ex) {
                ex.printStackTrace();
            }
        }
    }

    public static void main(String[] args) {
        launch(args);
    }
}
