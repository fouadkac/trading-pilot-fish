#property strict
#property indicator_chart_window

input int sma_period = 1000;   // Period for SMA
double sma_buffer[];

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   ArraySetAsSeries(sma_buffer, true); // Align newest data at index 0
   return INIT_SUCCEEDED;
  }

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
  }

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   static int count = 0;
   
   double tick_price = SymbolInfoDouble(_Symbol, SYMBOL_BID);  // get current tick price (Bid)

   // Shift array to right
   ArrayResize(sma_buffer, count + 1);
   for (int i = count; i > 0; i--)
      sma_buffer[i] = sma_buffer[i - 1];
   sma_buffer[0] = tick_price;
   count++;

   // Calculate SMA only when enough ticks collected
   if (count >= sma_period)
     {
      double sum = 0.0;
      for (int i = 0; i < sma_period; i++)
         sum += sma_buffer[i];
      double sma_value = sum / sma_period;

      // Print the SMA value
      Print("Current SMA(", sma_period, ") on Ticks: ", sma_value);
     }
  }
