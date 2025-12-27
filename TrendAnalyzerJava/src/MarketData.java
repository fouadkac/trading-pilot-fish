import java.time.LocalDateTime;

public class MarketData {
    public LocalDateTime time;
    public double open;
    public double high;
    public double low;
    public double close;

    public MarketData(LocalDateTime time, double open, double high, double low, double close) {
        this.time = time;
        this.open = open;
        this.high = high;
        this.low = low;
        this.close = close;
    }
}
