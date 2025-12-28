
#property indicator_chart_window
input double FakeHigh = 1.0;
input double FakeLow = 0.5;
input double FakeOpen = 0.8;
input double FakeClose = 0.75;
int OnInit()
{
   return(INIT_SUCCEEDED);
}
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime &time[],
                const double &open[],
                const double &high[],
                const double &low[],
                const double &close[],
                const long &tick_volume[],
                const long &volume[],
                const int &spread[])
{
   datetime fake_time = time[rates_total - 1] + PeriodSeconds();
   // Determine the coordinates of the rectangle for the body
   datetime body_left = (datetime)fake_time;
   double body_top = MathMax(FakeOpen, FakeClose);
   datetime body_right = (datetime)(fake_time + PeriodSeconds() / 1.5);
   double body_bottom = MathMin(FakeOpen, FakeClose);
   // Draw fake candlestick body
   ObjectCreate(0, "FakeCandleBody", OBJ_RECTANGLE, 0, body_left, body_top, body_right, body_bottom);
   ObjectSetInteger(0, "FakeCandleBody", OBJPROP_COLOR, clrGray);
   ObjectSetInteger(0, "FakeCandleBody", OBJPROP_STYLE, STYLE_SOLID);
   ObjectSetInteger(0, "FakeCandleBody", OBJPROP_FILL, clrGray);
   datetime wick_x = body_left + PeriodSeconds() / 3;
   // Draw fake candlestick wick at the bottom
   ObjectCreate(0, "FakeCandleWickBottom", OBJ_TREND, 0, wick_x, body_bottom, wick_x, FakeLow);
   ObjectSetInteger(0, "FakeCandleWickBottom", OBJPROP_COLOR, clrGray);
   ObjectSetInteger(0, "FakeCandleWickBottom", OBJPROP_STYLE, STYLE_SOLID);
   // Draw fake candlestick wick at the top
   ObjectCreate(0, "FakeCandleWickTop", OBJ_TREND, 0, wick_x, FakeHigh, wick_x, body_top);
   ObjectSetInteger(0, "FakeCandleWickTop", OBJPROP_COLOR, clrGray);
   ObjectSetInteger(0, "FakeCandleWickTop", OBJPROP_STYLE, STYLE_SOLID);
   // Draw horizontal line at FakeClose price
   ObjectCreate(0, "FakeCloseLine", OBJ_HLINE, 0, time[rates_total - 1], FakeClose);
   ObjectSetInteger(0, "FakeCloseLine", OBJPROP_COLOR, clrGray);
   ObjectSetInteger(0, "FakeCloseLine", OBJPROP_STYLE, STYLE_SOLID);
   return(rates_total);
}
void OnDeinit(const int reason)
{
   ObjectsDeleteAll(0, "FakeCandleBody");
   ObjectsDeleteAll(0, "FakeCandleWickTop");
   ObjectsDeleteAll(0, "FakeCandleWickBottom");
   ObjectsDeleteAll(0, "FakeCloseLine");
}
