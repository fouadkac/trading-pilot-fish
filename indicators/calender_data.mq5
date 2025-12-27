// Declare file handle
int handle;
bool testing = true;
datetime lastProcessedTime = 0; // To store the last processed time and avoid duplicates

// Arrays to store bid and ask prices for comparison
double bidArray[3];
double askArray[3];

//+------------------------------------------------------------------+
//| Expert initialization function                                 a   |
//+------------------------------------------------------------------+
int OnInit()
{
   // Open the file for writing (this file will be created on the user's disk)
   handle = FileOpen("test.csv", FILE_CSV | FILE_WRITE | FILE_ANSI, ";");
   if (handle < 1)
   {
      Print("File test.csv not found, the last error is ", GetLastError());
      return (INIT_FAILED);
   }
   else
   {
      Print("File opened successfully");
      // Write header to CSV
      FileWrite(handle, "Time", "Bid", "Ask", "Prev Bid", "Prev Ask", "FVG Type", "FVG Low", "FVG High");
   }

   return (INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                  |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   // Close the file when the Expert Advisor is removed
   FileClose(handle);
}

//+------------------------------------------------------------------+
//| Expert tester function for backtest data export                  |
//+------------------------------------------------------------------+
void OnTick()
{
   // Ensure we process data only when new time (to avoid repeating same data)
   if (TimeCurrent() != lastProcessedTime)
   {
      lastProcessedTime = TimeCurrent();

      // Retrieve the current bid and ask prices using SymbolInfo functions for backtest
      double currentBid = 0.0;
      double currentAsk = 0.0;

      if (SymbolInfoDouble(Symbol(), SYMBOL_BID, currentBid) && 
          SymbolInfoDouble(Symbol(), SYMBOL_ASK, currentAsk))
      {
         // Update bid and ask arrays
         bidArray[2] = bidArray[1];
         bidArray[1] = bidArray[0];
         bidArray[0] = currentBid;

         askArray[2] = askArray[1];
         askArray[1] = askArray[0];
         askArray[0] = currentAsk;

         // Get the previous bid/ask values from the arrays
         double prevBid = bidArray[1];
         double prevAsk = askArray[1];

         // Check for FVG up condition (Bid > Ask[1])
         string fvgType = "";
         double fvgLow = 0.0, fvgHigh = 0.0;

         if (currentBid > prevAsk) // FVG up
         {
            fvgType = "FVG Up";
            fvgLow = prevAsk;
            fvgHigh = currentBid;
         }
         else if (currentAsk < prevBid) // FVG down
         {
            fvgType = "FVG Down";
            fvgLow = prevBid;
            fvgHigh = currentAsk;
         }

         // If there is a valid FVG (either up or down), write to the file
         if (fvgType != "")
         {
            FileWrite(handle, 
               TimeToString(TimeCurrent(), TIME_DATE | TIME_SECONDS),
               currentBid, currentAsk, prevBid, prevAsk, // Include current and previous bid/ask prices
               fvgType, 
               fvgLow, fvgHigh
            );
         }
      }
      else
      {
         Print("Error retrieving bid/ask prices");
      }
   }
}
