//+------------------------------------------------------------------+
//|                     UTBot EA                                    |
//|   Uses UTBot indicator to open and close trades                |
//+------------------------------------------------------------------+
#include <Trade/Trade.mqh>

// Create an instance of CTrade
CTrade trade;

// Indicator parameters
input double AtrCoef = 2;   // ATR Coefficient (Sensitivity)
input int AtrLen = 1;       // ATR Period

// Buffers
double C1[], ATR[];
bool bull = false, bear = false;

// Indicator handle
int ATR_handle;

//+------------------------------------------------------------------+
//| Function to check the current position type                     |
//+------------------------------------------------------------------+
int GetCurrentPositionType()
{
   for (int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if (PositionSelect(i) && StringCompare(PositionGetString(POSITION_SYMBOL), Symbol()) == 0)
      {
         if (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) return 1;  // Buy position
         if (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL) return -1; // Sell position
      }
   }
   return 0; // No position
}

//+------------------------------------------------------------------+
//| Close all open positions                                        |
//+------------------------------------------------------------------+
bool CloseAllPositions()
{
   bool success = true;
   for (int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if (PositionSelect(i) && StringCompare(PositionGetString(POSITION_SYMBOL), Symbol()) == 0)
      {
         ulong ticket = PositionGetInteger(POSITION_TICKET);
         if (!trade.PositionClose(ticket))
         {
            Print("Error closing position: ", GetLastError());
            success = false;
         }
      }
   }
   return success; // Return true if all positions closed successfully
}

//+------------------------------------------------------------------+
//| Expert initialization function                                  |
//+------------------------------------------------------------------+
int OnInit()
{
   // Initialize ATR handle
   ATR_handle = iATR(NULL, 0, AtrLen);
   if (ATR_handle == INVALID_HANDLE)
   {
      Print("Error initializing ATR: ", GetLastError());
      return INIT_FAILED;
   }

   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert tick function                                            |
//+------------------------------------------------------------------+
void OnTick()
{
   // Prevent multiple positions
   if (PositionsTotal() > 0) return; 

   // Get ATR values
   if (CopyBuffer(ATR_handle, 0, 0, 1, ATR) <= 0)
   {
      Print("Error copying ATR values: ", GetLastError());
      return;
   }
   Print("ATR[0]: ", ATR[0]);

   // Resize arrays
   int limit = iBars(_Symbol, PERIOD_CURRENT) - 1;
   if (limit < 2) return; // Ensure enough bars exist

   ArrayResize(C1, limit + 2);

   bull = false;
   bear = false;

   // Calculate the trading signals
   for (int i = limit; i >= 1 && !IsStopped(); i--)
   {
      double loss = ATR[i] * AtrCoef;
      double t1 = iClose(Symbol(), 0, i) > C1[i + 1] ? iClose(Symbol(), 0, i) - loss : iClose(Symbol(), 0, i) + loss;
      double t2 = (iClose(Symbol(), 0, i) < C1[i + 1] && iClose(Symbol(), 0, i + 1) < C1[i + 1])
                    ? MathMin(C1[i + 1], iClose(Symbol(), 0, i) + loss)
                    : t1;
      C1[i] = (iClose(Symbol(), 0, i) > C1[i + 1] && iClose(Symbol(), 0, i + 1) > C1[i + 1])
                    ? MathMax(C1[i + 1], iClose(Symbol(), 0, i) - loss)
                    : t2;

      double h = MathAbs(iHigh(Symbol(), 0, i + 1) - iLow(Symbol(), 0, i + 1));

      if (iClose(Symbol(), 0, i) > C1[i] && iClose(Symbol(), 0, i + 1) <= C1[i + 1]) 
      {
         bull = true;
         bear = false;
      }
      if (iClose(Symbol(), 0, i) < C1[i] && iClose(Symbol(), 0, i + 1) >= C1[i + 1]) 
      {
         bear = true;
         bull = false;
      }
   }

   // Debugging information
   Print("Bull Signal: ", bull, " | Bear Signal: ", bear);

   int positionType = GetCurrentPositionType();
   Print("Current Position Type: ", positionType);

   if (bull && positionType == 0) 
   {
      Print("Opening Buy position...");
      if (!CloseAllPositions()) return;
      trade.Buy(0.01);  
   }
   else if (bear && positionType == 0)
   {
      Print("Opening Sell position...");
      if (!CloseAllPositions()) return;
      trade.Sell(0.01);  
   }

   // Test manuel : Ouvrir une position si rien ne se passe
   if (!bull && !bear)
   {
      Print("No valid signals. Testing manual buy...");
      trade.Buy(0.01);
   }
}  
