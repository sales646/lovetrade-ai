export type Json =
  | string
  | number
  | boolean
  | null
  | { [key: string]: Json | undefined }
  | Json[]

export type Database = {
  // Allows to automatically instantiate createClient with right options
  // instead of createClient<Database, { PostgrestVersion: 'XX' }>(URL, KEY)
  __InternalSupabase: {
    PostgrestVersion: "13.0.5"
  }
  public: {
    Tables: {
      bot_config: {
        Row: {
          alpaca_api_key: string | null
          alpaca_paper_trading: boolean | null
          alpaca_secret_key: string | null
          continuous_learning_enabled: boolean
          id: string
          is_active: boolean
          loop_interval_minutes: number
          loops_per_cycle: number
          max_concurrent_positions: number
          max_drawdown_pct: number
          max_position_size_pct: number
          risk_per_trade_pct: number
          updated_at: string
        }
        Insert: {
          alpaca_api_key?: string | null
          alpaca_paper_trading?: boolean | null
          alpaca_secret_key?: string | null
          continuous_learning_enabled?: boolean
          id?: string
          is_active?: boolean
          loop_interval_minutes?: number
          loops_per_cycle?: number
          max_concurrent_positions?: number
          max_drawdown_pct?: number
          max_position_size_pct?: number
          risk_per_trade_pct?: number
          updated_at?: string
        }
        Update: {
          alpaca_api_key?: string | null
          alpaca_paper_trading?: boolean | null
          alpaca_secret_key?: string | null
          continuous_learning_enabled?: boolean
          id?: string
          is_active?: boolean
          loop_interval_minutes?: number
          loops_per_cycle?: number
          max_concurrent_positions?: number
          max_drawdown_pct?: number
          max_position_size_pct?: number
          risk_per_trade_pct?: number
          updated_at?: string
        }
        Relationships: []
      }
      bot_loops: {
        Row: {
          completed_at: string | null
          error_message: string | null
          id: string
          loop_number: number
          positions_closed: number
          signals_generated: number
          started_at: string
          status: string
          total_pnl: number
          trades_placed: number
          trades_skipped: number
        }
        Insert: {
          completed_at?: string | null
          error_message?: string | null
          id?: string
          loop_number: number
          positions_closed?: number
          signals_generated?: number
          started_at?: string
          status?: string
          total_pnl?: number
          trades_placed?: number
          trades_skipped?: number
        }
        Update: {
          completed_at?: string | null
          error_message?: string | null
          id?: string
          loop_number?: number
          positions_closed?: number
          signals_generated?: number
          started_at?: string
          status?: string
          total_pnl?: number
          trades_placed?: number
          trades_skipped?: number
        }
        Relationships: []
      }
      expert_contributions: {
        Row: {
          accuracy: number
          created_at: string
          expert_name: string
          id: string
          loss_contribution: number
          sample_count: number
          training_metric_id: string | null
          weight: number
        }
        Insert: {
          accuracy: number
          created_at?: string
          expert_name: string
          id?: string
          loss_contribution: number
          sample_count: number
          training_metric_id?: string | null
          weight: number
        }
        Update: {
          accuracy?: number
          created_at?: string
          expert_name?: string
          id?: string
          loss_contribution?: number
          sample_count?: number
          training_metric_id?: string | null
          weight?: number
        }
        Relationships: [
          {
            foreignKeyName: "expert_contributions_training_metric_id_fkey"
            columns: ["training_metric_id"]
            isOneToOne: false
            referencedRelation: "rl_training_metrics"
            referencedColumns: ["id"]
          },
        ]
      }
      expert_trajectories: {
        Row: {
          action: number
          created_at: string
          delta_equity: number
          entry_quality: number | null
          fees: number
          id: string
          obs_features: Json
          regime_tag: string | null
          reward: number
          rr_ratio: number | null
          slippage: number
          symbol: string
          tactic_id: string
          timeframe: string
          timestamp: string
        }
        Insert: {
          action: number
          created_at?: string
          delta_equity: number
          entry_quality?: number | null
          fees: number
          id?: string
          obs_features: Json
          regime_tag?: string | null
          reward: number
          rr_ratio?: number | null
          slippage: number
          symbol: string
          tactic_id: string
          timeframe: string
          timestamp: string
        }
        Update: {
          action?: number
          created_at?: string
          delta_equity?: number
          entry_quality?: number | null
          fees?: number
          id?: string
          obs_features?: Json
          regime_tag?: string | null
          reward?: number
          rr_ratio?: number | null
          slippage?: number
          symbol?: string
          tactic_id?: string
          timeframe?: string
          timestamp?: string
        }
        Relationships: []
      }
      historical_bars: {
        Row: {
          close: number
          created_at: string
          high: number
          id: string
          low: number
          open: number
          symbol: string
          timeframe: string
          timestamp: string
          volume: number
        }
        Insert: {
          close: number
          created_at?: string
          high: number
          id?: string
          low: number
          open: number
          symbol: string
          timeframe: string
          timestamp: string
          volume: number
        }
        Update: {
          close?: number
          created_at?: string
          high?: number
          id?: string
          low?: number
          open?: number
          symbol?: string
          timeframe?: string
          timestamp?: string
          volume?: number
        }
        Relationships: []
      }
      news_features: {
        Row: {
          article_snippet: string | null
          created_at: string
          freshness_minutes: number | null
          headline: string
          id: string
          relevance_score: number | null
          sentiment: number | null
          source: string | null
          surprise_score: number | null
          symbol: string
          timestamp: string
        }
        Insert: {
          article_snippet?: string | null
          created_at?: string
          freshness_minutes?: number | null
          headline: string
          id?: string
          relevance_score?: number | null
          sentiment?: number | null
          source?: string | null
          surprise_score?: number | null
          symbol: string
          timestamp: string
        }
        Update: {
          article_snippet?: string | null
          created_at?: string
          freshness_minutes?: number | null
          headline?: string
          id?: string
          relevance_score?: number | null
          sentiment?: number | null
          source?: string | null
          surprise_score?: number | null
          symbol?: string
          timestamp?: string
        }
        Relationships: []
      }
      online_learning: {
        Row: {
          created_at: string
          id: string
          learning_rate: number
          loss: number | null
          metrics: Json | null
          model_type: string
          samples_processed: number
        }
        Insert: {
          created_at?: string
          id?: string
          learning_rate: number
          loss?: number | null
          metrics?: Json | null
          model_type: string
          samples_processed: number
        }
        Update: {
          created_at?: string
          id?: string
          learning_rate?: number
          loss?: number | null
          metrics?: Json | null
          model_type?: string
          samples_processed?: number
        }
        Relationships: []
      }
      position_sizing: {
        Row: {
          base_size: number
          created_at: string
          drawdown_adjusted_size: number
          factors: Json
          final_size: number
          id: string
          risk_amount: number
          rl_adjusted_size: number
          signal_id: string | null
          volatility_adjusted_size: number
        }
        Insert: {
          base_size: number
          created_at?: string
          drawdown_adjusted_size: number
          factors: Json
          final_size: number
          id?: string
          risk_amount: number
          rl_adjusted_size: number
          signal_id?: string | null
          volatility_adjusted_size: number
        }
        Update: {
          base_size?: number
          created_at?: string
          drawdown_adjusted_size?: number
          factors?: Json
          final_size?: number
          id?: string
          risk_amount?: number
          rl_adjusted_size?: number
          signal_id?: string | null
          volatility_adjusted_size?: number
        }
        Relationships: [
          {
            foreignKeyName: "position_sizing_signal_id_fkey"
            columns: ["signal_id"]
            isOneToOne: false
            referencedRelation: "trading_signals"
            referencedColumns: ["id"]
          },
        ]
      }
      positions: {
        Row: {
          current_price: number | null
          entry_price: number
          id: string
          opened_at: string
          side: string
          size: number
          stop_loss: number | null
          symbol: string
          take_profit: number | null
          unrealized_pnl: number | null
          updated_at: string
        }
        Insert: {
          current_price?: number | null
          entry_price: number
          id?: string
          opened_at?: string
          side: string
          size: number
          stop_loss?: number | null
          symbol: string
          take_profit?: number | null
          unrealized_pnl?: number | null
          updated_at?: string
        }
        Update: {
          current_price?: number | null
          entry_price?: number
          id?: string
          opened_at?: string
          side?: string
          size?: number
          stop_loss?: number | null
          symbol?: string
          take_profit?: number | null
          unrealized_pnl?: number | null
          updated_at?: string
        }
        Relationships: []
      }
      risk_assessments: {
        Row: {
          adjusted_size: number
          assessed_at: string
          factors: Json
          id: string
          reason: string
          risk_score: number
          should_execute: boolean
          signal_id: string | null
        }
        Insert: {
          adjusted_size: number
          assessed_at?: string
          factors: Json
          id?: string
          reason: string
          risk_score: number
          should_execute: boolean
          signal_id?: string | null
        }
        Update: {
          adjusted_size?: number
          assessed_at?: string
          factors?: Json
          id?: string
          reason?: string
          risk_score?: number
          should_execute?: boolean
          signal_id?: string | null
        }
        Relationships: [
          {
            foreignKeyName: "risk_assessments_signal_id_fkey"
            columns: ["signal_id"]
            isOneToOne: false
            referencedRelation: "trading_signals"
            referencedColumns: ["id"]
          },
        ]
      }
      rl_decisions: {
        Row: {
          action: string
          confidence: number
          created_at: string
          id: string
          q_value: number | null
          reasoning: string | null
          signal_id: string | null
          state_features: Json
        }
        Insert: {
          action: string
          confidence: number
          created_at?: string
          id?: string
          q_value?: number | null
          reasoning?: string | null
          signal_id?: string | null
          state_features: Json
        }
        Update: {
          action?: string
          confidence?: number
          created_at?: string
          id?: string
          q_value?: number | null
          reasoning?: string | null
          signal_id?: string | null
          state_features?: Json
        }
        Relationships: [
          {
            foreignKeyName: "rl_decisions_signal_id_fkey"
            columns: ["signal_id"]
            isOneToOne: false
            referencedRelation: "trading_signals"
            referencedColumns: ["id"]
          },
        ]
      }
      rl_q_state: {
        Row: {
          alpha: number
          episode_count: number
          epsilon: number
          gamma: number
          id: number
          q_table: Json
          updated_at: string
        }
        Insert: {
          alpha?: number
          episode_count?: number
          epsilon?: number
          gamma?: number
          id?: number
          q_table?: Json
          updated_at?: string
        }
        Update: {
          alpha?: number
          episode_count?: number
          epsilon?: number
          gamma?: number
          id?: number
          q_table?: Json
          updated_at?: string
        }
        Relationships: []
      }
      rl_training_metrics: {
        Row: {
          account_equity: number | null
          action_buy_pct: number | null
          action_hold_pct: number | null
          action_sell_pct: number | null
          alpha_mix: number | null
          avg_dollar_pnl: number | null
          avg_return_pct: number | null
          avg_reward: number
          avg_steps: number
          created_at: string
          duration_seconds: number
          episodes: number
          epsilon: number
          expert_accuracies: Json | null
          id: string
          l_imitation: number | null
          l_rl: number | null
          l_total: number | null
          q_table_size: number
          sharpe_ratio: number | null
          sortino_ratio: number | null
          total_episodes: number
          total_trades: number | null
          win_rate_pct: number | null
          winning_trades: number | null
        }
        Insert: {
          account_equity?: number | null
          action_buy_pct?: number | null
          action_hold_pct?: number | null
          action_sell_pct?: number | null
          alpha_mix?: number | null
          avg_dollar_pnl?: number | null
          avg_return_pct?: number | null
          avg_reward: number
          avg_steps: number
          created_at?: string
          duration_seconds: number
          episodes: number
          epsilon: number
          expert_accuracies?: Json | null
          id?: string
          l_imitation?: number | null
          l_rl?: number | null
          l_total?: number | null
          q_table_size: number
          sharpe_ratio?: number | null
          sortino_ratio?: number | null
          total_episodes: number
          total_trades?: number | null
          win_rate_pct?: number | null
          winning_trades?: number | null
        }
        Update: {
          account_equity?: number | null
          action_buy_pct?: number | null
          action_hold_pct?: number | null
          action_sell_pct?: number | null
          alpha_mix?: number | null
          avg_dollar_pnl?: number | null
          avg_return_pct?: number | null
          avg_reward?: number
          avg_steps?: number
          created_at?: string
          duration_seconds?: number
          episodes?: number
          epsilon?: number
          expert_accuracies?: Json | null
          id?: string
          l_imitation?: number | null
          l_rl?: number | null
          l_total?: number | null
          q_table_size?: number
          sharpe_ratio?: number | null
          sortino_ratio?: number | null
          total_episodes?: number
          total_trades?: number | null
          win_rate_pct?: number | null
          winning_trades?: number | null
        }
        Relationships: []
      }
      signal_correlations: {
        Row: {
          correlation: number
          created_at: string
          id: string
          position_symbol: string
          signal_id: string | null
        }
        Insert: {
          correlation: number
          created_at?: string
          id?: string
          position_symbol: string
          signal_id?: string | null
        }
        Update: {
          correlation?: number
          created_at?: string
          id?: string
          position_symbol?: string
          signal_id?: string | null
        }
        Relationships: [
          {
            foreignKeyName: "signal_correlations_signal_id_fkey"
            columns: ["signal_id"]
            isOneToOne: false
            referencedRelation: "trading_signals"
            referencedColumns: ["id"]
          },
        ]
      }
      strategy_performance: {
        Row: {
          avg_rr_ratio: number
          created_at: string
          id: string
          is_active: boolean
          last_trade_at: string | null
          losing_trades: number
          max_drawdown: number
          profit_factor: number
          sharpe_ratio: number
          strategy_name: string
          total_trades: number
          updated_at: string
          win_rate: number
          winning_trades: number
        }
        Insert: {
          avg_rr_ratio?: number
          created_at?: string
          id?: string
          is_active?: boolean
          last_trade_at?: string | null
          losing_trades?: number
          max_drawdown?: number
          profit_factor?: number
          sharpe_ratio?: number
          strategy_name: string
          total_trades?: number
          updated_at?: string
          win_rate?: number
          winning_trades?: number
        }
        Update: {
          avg_rr_ratio?: number
          created_at?: string
          id?: string
          is_active?: boolean
          last_trade_at?: string | null
          losing_trades?: number
          max_drawdown?: number
          profit_factor?: number
          sharpe_ratio?: number
          strategy_name?: string
          total_trades?: number
          updated_at?: string
          win_rate?: number
          winning_trades?: number
        }
        Relationships: []
      }
      symbols: {
        Row: {
          created_at: string
          exchange: string | null
          industry: string | null
          is_active: boolean | null
          last_fetched: string | null
          name: string | null
          sector: string | null
          symbol: string
          updated_at: string
        }
        Insert: {
          created_at?: string
          exchange?: string | null
          industry?: string | null
          is_active?: boolean | null
          last_fetched?: string | null
          name?: string | null
          sector?: string | null
          symbol: string
          updated_at?: string
        }
        Update: {
          created_at?: string
          exchange?: string | null
          industry?: string | null
          is_active?: boolean | null
          last_fetched?: string | null
          name?: string | null
          sector?: string | null
          symbol?: string
          updated_at?: string
        }
        Relationships: []
      }
      system_logs: {
        Row: {
          created_at: string
          id: string
          level: string
          message: string
          metadata: Json | null
          source: string
        }
        Insert: {
          created_at?: string
          id?: string
          level: string
          message: string
          metadata?: Json | null
          source: string
        }
        Update: {
          created_at?: string
          id?: string
          level?: string
          message?: string
          metadata?: Json | null
          source?: string
        }
        Relationships: []
      }
      technical_indicators: {
        Row: {
          atr_14: number | null
          created_at: string
          ema_20: number | null
          ema_50: number | null
          id: string
          intraday_position: number | null
          range_pct: number | null
          rsi_14: number | null
          symbol: string
          timeframe: string
          timestamp: string
          volume_zscore: number | null
          vwap: number | null
          vwap_distance_pct: number | null
        }
        Insert: {
          atr_14?: number | null
          created_at?: string
          ema_20?: number | null
          ema_50?: number | null
          id?: string
          intraday_position?: number | null
          range_pct?: number | null
          rsi_14?: number | null
          symbol: string
          timeframe: string
          timestamp: string
          volume_zscore?: number | null
          vwap?: number | null
          vwap_distance_pct?: number | null
        }
        Update: {
          atr_14?: number | null
          created_at?: string
          ema_20?: number | null
          ema_50?: number | null
          id?: string
          intraday_position?: number | null
          range_pct?: number | null
          rsi_14?: number | null
          symbol?: string
          timeframe?: string
          timestamp?: string
          volume_zscore?: number | null
          vwap?: number | null
          vwap_distance_pct?: number | null
        }
        Relationships: []
      }
      trades: {
        Row: {
          action: string
          closed_at: string | null
          entry_price: number
          executed_at: string
          exit_price: number | null
          id: string
          pnl: number | null
          risk_assessment_id: string | null
          signal_id: string | null
          size: number
          status: string
          stop_loss: number | null
          symbol: string
          take_profit: number | null
        }
        Insert: {
          action: string
          closed_at?: string | null
          entry_price: number
          executed_at?: string
          exit_price?: number | null
          id?: string
          pnl?: number | null
          risk_assessment_id?: string | null
          signal_id?: string | null
          size: number
          status?: string
          stop_loss?: number | null
          symbol: string
          take_profit?: number | null
        }
        Update: {
          action?: string
          closed_at?: string | null
          entry_price?: number
          executed_at?: string
          exit_price?: number | null
          id?: string
          pnl?: number | null
          risk_assessment_id?: string | null
          signal_id?: string | null
          size?: number
          status?: string
          stop_loss?: number | null
          symbol?: string
          take_profit?: number | null
        }
        Relationships: [
          {
            foreignKeyName: "trades_risk_assessment_id_fkey"
            columns: ["risk_assessment_id"]
            isOneToOne: false
            referencedRelation: "risk_assessments"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "trades_signal_id_fkey"
            columns: ["signal_id"]
            isOneToOne: false
            referencedRelation: "trading_signals"
            referencedColumns: ["id"]
          },
        ]
      }
      trading_signals: {
        Row: {
          action: string
          confidence: number
          created_at: string
          id: string
          market_data: Json
          proposed_size: number
          source: string
          symbol: string
        }
        Insert: {
          action: string
          confidence: number
          created_at?: string
          id?: string
          market_data: Json
          proposed_size: number
          source: string
          symbol: string
        }
        Update: {
          action?: string
          confidence?: number
          created_at?: string
          id?: string
          market_data?: Json
          proposed_size?: number
          source?: string
          symbol?: string
        }
        Relationships: []
      }
      training_metrics: {
        Row: {
          action_buy_pct: number | null
          action_hold_pct: number | null
          action_sell_pct: number | null
          avg_rr: number | null
          created_at: string
          entropy: number | null
          epoch: number
          good_skips: number | null
          id: string
          max_drawdown: number | null
          mean_reward: number | null
          missed_winners: number | null
          policy_loss: number | null
          profit_factor: number | null
          run_id: string
          sharpe_ratio: number | null
          split: string
          value_loss: number | null
          win_rate: number | null
        }
        Insert: {
          action_buy_pct?: number | null
          action_hold_pct?: number | null
          action_sell_pct?: number | null
          avg_rr?: number | null
          created_at?: string
          entropy?: number | null
          epoch: number
          good_skips?: number | null
          id?: string
          max_drawdown?: number | null
          mean_reward?: number | null
          missed_winners?: number | null
          policy_loss?: number | null
          profit_factor?: number | null
          run_id: string
          sharpe_ratio?: number | null
          split: string
          value_loss?: number | null
          win_rate?: number | null
        }
        Update: {
          action_buy_pct?: number | null
          action_hold_pct?: number | null
          action_sell_pct?: number | null
          avg_rr?: number | null
          created_at?: string
          entropy?: number | null
          epoch?: number
          good_skips?: number | null
          id?: string
          max_drawdown?: number | null
          mean_reward?: number | null
          missed_winners?: number | null
          policy_loss?: number | null
          profit_factor?: number | null
          run_id?: string
          sharpe_ratio?: number | null
          split?: string
          value_loss?: number | null
          win_rate?: number | null
        }
        Relationships: [
          {
            foreignKeyName: "training_metrics_run_id_fkey"
            columns: ["run_id"]
            isOneToOne: false
            referencedRelation: "training_runs"
            referencedColumns: ["id"]
          },
        ]
      }
      training_runs: {
        Row: {
          best_checkpoint_path: string | null
          best_val_sharpe: number | null
          completed_at: string | null
          created_at: string
          current_epoch: number | null
          hyperparams: Json
          id: string
          phase: string
          run_name: string
          started_at: string
          status: string
          total_epochs: number | null
        }
        Insert: {
          best_checkpoint_path?: string | null
          best_val_sharpe?: number | null
          completed_at?: string | null
          created_at?: string
          current_epoch?: number | null
          hyperparams: Json
          id?: string
          phase: string
          run_name: string
          started_at?: string
          status?: string
          total_epochs?: number | null
        }
        Update: {
          best_checkpoint_path?: string | null
          best_val_sharpe?: number | null
          completed_at?: string | null
          created_at?: string
          current_epoch?: number | null
          hyperparams?: Json
          id?: string
          phase?: string
          run_name?: string
          started_at?: string
          status?: string
          total_epochs?: number | null
        }
        Relationships: []
      }
    }
    Views: {
      [_ in never]: never
    }
    Functions: {
      [_ in never]: never
    }
    Enums: {
      [_ in never]: never
    }
    CompositeTypes: {
      [_ in never]: never
    }
  }
}

type DatabaseWithoutInternals = Omit<Database, "__InternalSupabase">

type DefaultSchema = DatabaseWithoutInternals[Extract<keyof Database, "public">]

export type Tables<
  DefaultSchemaTableNameOrOptions extends
    | keyof (DefaultSchema["Tables"] & DefaultSchema["Views"])
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof (DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"] &
        DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Views"])
    : never = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? (DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"] &
      DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Views"])[TableName] extends {
      Row: infer R
    }
    ? R
    : never
  : DefaultSchemaTableNameOrOptions extends keyof (DefaultSchema["Tables"] &
        DefaultSchema["Views"])
    ? (DefaultSchema["Tables"] &
        DefaultSchema["Views"])[DefaultSchemaTableNameOrOptions] extends {
        Row: infer R
      }
      ? R
      : never
    : never

export type TablesInsert<
  DefaultSchemaTableNameOrOptions extends
    | keyof DefaultSchema["Tables"]
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"]
    : never = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"][TableName] extends {
      Insert: infer I
    }
    ? I
    : never
  : DefaultSchemaTableNameOrOptions extends keyof DefaultSchema["Tables"]
    ? DefaultSchema["Tables"][DefaultSchemaTableNameOrOptions] extends {
        Insert: infer I
      }
      ? I
      : never
    : never

export type TablesUpdate<
  DefaultSchemaTableNameOrOptions extends
    | keyof DefaultSchema["Tables"]
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"]
    : never = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"][TableName] extends {
      Update: infer U
    }
    ? U
    : never
  : DefaultSchemaTableNameOrOptions extends keyof DefaultSchema["Tables"]
    ? DefaultSchema["Tables"][DefaultSchemaTableNameOrOptions] extends {
        Update: infer U
      }
      ? U
      : never
    : never

export type Enums<
  DefaultSchemaEnumNameOrOptions extends
    | keyof DefaultSchema["Enums"]
    | { schema: keyof DatabaseWithoutInternals },
  EnumName extends DefaultSchemaEnumNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaEnumNameOrOptions["schema"]]["Enums"]
    : never = never,
> = DefaultSchemaEnumNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[DefaultSchemaEnumNameOrOptions["schema"]]["Enums"][EnumName]
  : DefaultSchemaEnumNameOrOptions extends keyof DefaultSchema["Enums"]
    ? DefaultSchema["Enums"][DefaultSchemaEnumNameOrOptions]
    : never

export type CompositeTypes<
  PublicCompositeTypeNameOrOptions extends
    | keyof DefaultSchema["CompositeTypes"]
    | { schema: keyof DatabaseWithoutInternals },
  CompositeTypeName extends PublicCompositeTypeNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[PublicCompositeTypeNameOrOptions["schema"]]["CompositeTypes"]
    : never = never,
> = PublicCompositeTypeNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[PublicCompositeTypeNameOrOptions["schema"]]["CompositeTypes"][CompositeTypeName]
  : PublicCompositeTypeNameOrOptions extends keyof DefaultSchema["CompositeTypes"]
    ? DefaultSchema["CompositeTypes"][PublicCompositeTypeNameOrOptions]
    : never

export const Constants = {
  public: {
    Enums: {},
  },
} as const
