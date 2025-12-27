string address = "localhost";
int port = 8466;
int socket;
bool isConnected = false;
long previousVolume = 0;

input double inpDivisor = 2;
input ENUM_APPLIED_PRICE inpPrice = PRICE_CLOSE;
input int hmaPeriod250 = 250;
input int hmaPeriod25 = 25;

int hma_handle_250;
int hma_handle_25;

double hma_values_250[];
double hma_values_25[];

int OnInit()
{
    // Initialize HMA handles
    hma_handle_250 = iCustom(_Symbol, PERIOD_CURRENT, "HMA_data.ex5", inpDivisor, inpPrice, hmaPeriod250);
    hma_handle_25 = iCustom(_Symbol, PERIOD_CURRENT, "HMA_data.ex5", inpDivisor, inpPrice, hmaPeriod25);

    if (hma_handle_250 == INVALID_HANDLE || hma_handle_25 == INVALID_HANDLE)
    {
        Print("Error: Failed to initialize HMA handles.");
        return INIT_FAILED;
    }

    // Initialize socket
    socket = SocketCreate();
    if (socket == INVALID_HANDLE)
    {
        Print("Error: SocketCreate failure. ", GetLastError());
        return INIT_FAILED;
    }

    if (SocketConnect(socket, address, port, 10000))
    {
        Print("[INFO]\tConnection established");
        isConnected = true;
        return INIT_SUCCEEDED;
    }
    else
    {
        Print("Error: SocketConnect failure. ", GetLastError());
        SocketClose(socket);
        return INIT_FAILED;
    }
}

void SendHistoricalDataWithDetails()
{
    if (!isConnected)
    {
        Print("Error: Socket is not connected.");
        return;
    }

    // Calculate the start time (three months ago)
    datetime currentTime = TimeCurrent();
    datetime startTime = currentTime - (90 * 24 * 60 * 60); // 90 days

    // Retrieve historical data
    MqlRates ratesArray[];
    if (CopyRates(_Symbol, PERIOD_M1, startTime, currentTime, ratesArray) <= 0)
    {
        Print("Error: Failed to copy historical data. ", GetLastError());
        return;
    }

    // Retrieve HMA values for historical data
    ArrayResize(hma_values_250, ArraySize(ratesArray));
    ArrayResize(hma_values_25, ArraySize(ratesArray));

    if (CopyBuffer(hma_handle_250, 0, 0, ArraySize(ratesArray), hma_values_250) < 0 ||
        CopyBuffer(hma_handle_25, 0, 0, ArraySize(ratesArray), hma_values_25) < 0)
    {
        Print("Error: Failed to copy HMA data.");
        return;
    }

    // Process each bar
    for (int i = ArraySize(ratesArray) - 1; i >= 0; i--)
    {
        double hma250 = hma_values_250[i];
        double hma25 = hma_values_25[i];

        // Bid, ask, and spread
        double bidPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);
        double askPrice = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
        double spread = (askPrice - bidPrice) * 10;

        // Create the message string
        string message = "Symbol: " + _Symbol +
                         ", Time: " + TimeToString(ratesArray[i].time, TIME_DATE | TIME_SECONDS) +
                         ", Bid Price: " + DoubleToString(bidPrice, 5) +
                         ", Ask Price: " + DoubleToString(askPrice, 5) +
                         ", Spread: " + DoubleToString(spread, 5) +
                         ", Volume: " + DoubleToString(ratesArray[i].tick_volume, 0) +
                         ", HMA25: " + DoubleToString(hma25, 5) +
                         ", HMA250: " + DoubleToString(hma250, 5) +
                         ", Close: " + DoubleToString(ratesArray[i].close, 5) +
                         ", Open: " + DoubleToString(ratesArray[i].open, 5);

        // Convert message to char array
        char req[];
        int len = StringToCharArray(message, req) - 1;

        // Send the data over the socket
        if (!SocketSend(socket, req, len))
        {
            Print("Error: Failed to send data. ", GetLastError());
            break;
        }
        else
        {
            Print("[INFO]\tData sent: ", message);
        }
    }
}

void OnDeinit(const int reason)
{
    // Close the socket connection
    if (isConnected)
    {
        SocketClose(socket);
        Print("[INFO]\tSocket closed");
    }
}
