import { supabase } from "@/integrations/supabase/client";

export async function triggerDataPopulation() {
  try {
    console.log('🚀 Triggering historical data population...');
    
    const { data, error } = await supabase.functions.invoke('populate-historical-data');
    
    if (error) {
      console.error('Failed to trigger data population:', error);
      throw error;
    }
    
    console.log('✅ Data population started:', data);
    return data;
  } catch (error) {
    console.error('Error triggering data population:', error);
    throw error;
  }
}

export async function getHistoricalBarCount() {
  try {
    const { count, error } = await supabase
      .from('historical_bars')
      .select('*', { count: 'exact', head: true });
    
    if (error) throw error;
    
    return count || 0;
  } catch (error) {
    console.error('Error getting bar count:', error);
    return 0;
  }
}

export async function checkDataPopulationStatus() {
  try {
    const { data, error } = await supabase
      .from('system_logs')
      .select('*')
      .eq('source', 'populate-historical-data')
      .order('created_at', { ascending: false })
      .limit(1);
    
    if (error) throw error;
    
    return data?.[0] || null;
  } catch (error) {
    console.error('Error checking status:', error);
    return null;
  }
}
