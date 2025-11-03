import { createClient } from 'https://esm.sh/@supabase/supabase-js@2.39.3'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

interface YahooBar {
  date: number
  open: number
  high: number
  low: number
  close: number
  volume: number
}

Deno.serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders })
  }

  const supabaseUrl = Deno.env.get('SUPABASE_URL')!
  const supabaseKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!
  const supabase = createClient(supabaseUrl, supabaseKey)

  // Symbols to fetch data for
  const symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'AMZN', 'META', 'JPM', 'BAC', 'WMT']
  const timeframes = ['1m', '5m', '1h', '1d']
  
  console.log('🚀 Starting bulk historical data fetch from Yahoo Finance...')
  console.log(`📊 Symbols: ${symbols.join(', ')}`)
  console.log(`⏱️  Timeframes: ${timeframes.join(', ')}`)

  // Run the data fetch as a background task
  const fetchTask = async () => {
    let totalBarsInserted = 0
    const results = []

    try {

    for (const symbol of symbols) {
      for (const timeframe of timeframes) {
        try {
          console.log(`\n📥 Fetching ${symbol} ${timeframe}...`)
          
          // Determine period and interval for Yahoo Finance
          let period = '5y' // 5 years
          let interval = timeframe
          
          // Yahoo Finance API call
          const yahooUrl = `https://query1.finance.yahoo.com/v8/finance/chart/${symbol}?period1=0&period2=9999999999&interval=${interval}`
          
          const response = await fetch(yahooUrl, {
            headers: {
              'User-Agent': 'Mozilla/5.0'
            }
          })

          if (!response.ok) {
            console.error(`❌ Failed to fetch ${symbol} ${timeframe}: ${response.status}`)
            continue
          }

          const data = await response.json()
          
          if (!data.chart?.result?.[0]?.timestamp) {
            console.error(`❌ No data returned for ${symbol} ${timeframe}`)
            continue
          }

          const result = data.chart.result[0]
          const timestamps = result.timestamp
          const quotes = result.indicators.quote[0]

          // Transform to our format
          const bars = timestamps.map((ts: number, idx: number) => ({
            symbol,
            timeframe,
            timestamp: new Date(ts * 1000).toISOString(),
            open: quotes.open[idx],
            high: quotes.high[idx],
            low: quotes.low[idx],
            close: quotes.close[idx],
            volume: quotes.volume[idx],
          })).filter((bar: any) => 
            bar.open && bar.high && bar.low && bar.close && bar.volume
          )

          console.log(`   📊 Processing ${bars.length} bars...`)

          // Insert in batches of 1000
          const batchSize = 1000
          let inserted = 0
          
          for (let i = 0; i < bars.length; i += batchSize) {
            const batch = bars.slice(i, i + batchSize)
            
            const { error } = await supabase
              .from('historical_bars')
              .upsert(batch, {
                onConflict: 'symbol,timeframe,timestamp',
                ignoreDuplicates: true
              })

            if (error) {
              console.error(`   ⚠️  Batch insert error: ${error.message}`)
            } else {
              inserted += batch.length
            }
          }

          totalBarsInserted += inserted
          console.log(`   ✅ Inserted ${inserted} bars for ${symbol} ${timeframe}`)
          
          results.push({
            symbol,
            timeframe,
            bars_fetched: bars.length,
            bars_inserted: inserted
          })

          // Small delay to avoid rate limiting
          await new Promise(resolve => setTimeout(resolve, 200))

        } catch (error) {
          console.error(`❌ Error processing ${symbol} ${timeframe}:`, error)
          results.push({
            symbol,
            timeframe,
            error: error instanceof Error ? error.message : String(error)
          })
        }
      }
    }

      console.log(`\n✅ Completed! Total bars inserted: ${totalBarsInserted}`)

      // Log final results to Supabase
      await supabase.from('system_logs').insert({
        source: 'populate-historical-data',
        level: 'info',
        message: `Completed data population: ${totalBarsInserted} bars inserted`,
        metadata: { results }
      })

    } catch (error) {
      console.error('❌ Fatal error:', error)
      await supabase.from('system_logs').insert({
        source: 'populate-historical-data',
        level: 'error',
        message: error instanceof Error ? error.message : String(error)
      })
    }
  }

  // Start background task (no await - let it run)
  fetchTask().catch(error => {
    console.error('Background task error:', error)
  })

  // Return immediate response
  return new Response(
    JSON.stringify({
      success: true,
      message: 'Data population started in background',
      status: 'running'
    }),
    { headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
  )
})
