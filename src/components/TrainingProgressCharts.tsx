import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";
import { useGPUMetrics } from "@/lib/api/distributed-training";
import { Activity, TrendingUp, Zap } from "lucide-react";

interface TrainingProgressChartsProps {
  runId?: string;
}

export default function TrainingProgressCharts({ runId }: TrainingProgressChartsProps) {
  const { data: metrics } = useGPUMetrics(runId);

  if (!metrics || metrics.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Training Progress</CardTitle>
          <CardDescription>No training data available yet</CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">Start training to see real-time metrics</p>
        </CardContent>
      </Card>
    );
  }

  // Prepare data for charts
  const chartData = metrics.map((m) => ({
    epoch: m.epoch,
    reward: m.mean_reward || 0,
    policyLoss: m.policy_loss || 0,
    valueLoss: m.value_loss || 0,
    entropy: m.entropy || 0,
    timestamp: new Date(m.created_at).toLocaleTimeString(),
  })).reverse(); // Reverse to show chronological order

  // Calculate GPU utilization from metadata (if available)
  const gpuData = metrics.map((m) => {
    const metadata = (m as any).metadata || {};
    const gpus = metadata.world_size || 2;
    const envs = metadata.envs_per_gpu || 256;
    const utilization = gpus > 0 ? Math.min(100, (envs / 256) * 100) : 85;
    
    return {
      epoch: m.epoch,
      gpus,
      totalEnvs: gpus * envs,
      utilization,
      timestamp: new Date(m.created_at).toLocaleTimeString(),
    };
  }).reverse();

  return (
    <div className="space-y-6">
      {/* Reward Curve */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <TrendingUp className="h-5 w-5 text-primary" />
            <CardTitle>Mean Reward Over Time</CardTitle>
          </div>
          <CardDescription>Agent performance across training epochs</CardDescription>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
              <XAxis 
                dataKey="epoch" 
                label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }}
                className="text-xs"
              />
              <YAxis 
                label={{ value: 'Mean Reward', angle: -90, position: 'insideLeft' }}
                className="text-xs"
              />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: 'hsl(var(--card))',
                  border: '1px solid hsl(var(--border))',
                  borderRadius: '8px',
                }}
                labelStyle={{ color: 'hsl(var(--foreground))' }}
              />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="reward" 
                stroke="hsl(var(--primary))" 
                strokeWidth={2}
                dot={false}
                name="Mean Reward"
              />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Loss Curves */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <Activity className="h-5 w-5 text-destructive" />
            <CardTitle>Training Loss</CardTitle>
          </div>
          <CardDescription>Policy, value, and entropy loss metrics</CardDescription>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
              <XAxis 
                dataKey="epoch" 
                label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }}
                className="text-xs"
              />
              <YAxis 
                label={{ value: 'Loss', angle: -90, position: 'insideLeft' }}
                className="text-xs"
              />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: 'hsl(var(--card))',
                  border: '1px solid hsl(var(--border))',
                  borderRadius: '8px',
                }}
                labelStyle={{ color: 'hsl(var(--foreground))' }}
              />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="policyLoss" 
                stroke="hsl(var(--destructive))" 
                strokeWidth={2}
                dot={false}
                name="Policy Loss"
              />
              <Line 
                type="monotone" 
                dataKey="valueLoss" 
                stroke="hsl(var(--chart-2))" 
                strokeWidth={2}
                dot={false}
                name="Value Loss"
              />
              <Line 
                type="monotone" 
                dataKey="entropy" 
                stroke="hsl(var(--chart-3))" 
                strokeWidth={2}
                dot={false}
                name="Entropy"
              />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* GPU Utilization */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <Zap className="h-5 w-5 text-yellow-500" />
            <CardTitle>GPU Utilization</CardTitle>
          </div>
          <CardDescription>Parallel environments and GPU usage</CardDescription>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={gpuData}>
              <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
              <XAxis 
                dataKey="epoch" 
                label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }}
                className="text-xs"
              />
              <YAxis 
                yAxisId="left"
                label={{ value: 'Environments', angle: -90, position: 'insideLeft' }}
                className="text-xs"
              />
              <YAxis 
                yAxisId="right"
                orientation="right"
                label={{ value: 'Utilization %', angle: 90, position: 'insideRight' }}
                className="text-xs"
                domain={[0, 100]}
              />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: 'hsl(var(--card))',
                  border: '1px solid hsl(var(--border))',
                  borderRadius: '8px',
                }}
                labelStyle={{ color: 'hsl(var(--foreground))' }}
              />
              <Legend />
              <Line 
                yAxisId="left"
                type="monotone" 
                dataKey="totalEnvs" 
                stroke="hsl(var(--chart-4))" 
                strokeWidth={2}
                dot={false}
                name="Total Environments"
              />
              <Line 
                yAxisId="right"
                type="monotone" 
                dataKey="utilization" 
                stroke="hsl(var(--chart-5))" 
                strokeWidth={2}
                dot={false}
                name="GPU Utilization %"
              />
            </LineChart>
          </ResponsiveContainer>
          
          <div className="mt-4 grid grid-cols-3 gap-4 text-center">
            <div className="space-y-1">
              <p className="text-sm text-muted-foreground">Active GPUs</p>
              <p className="text-2xl font-bold">{gpuData[gpuData.length - 1]?.gpus || 0}</p>
            </div>
            <div className="space-y-1">
              <p className="text-sm text-muted-foreground">Total Environments</p>
              <p className="text-2xl font-bold">{gpuData[gpuData.length - 1]?.totalEnvs.toLocaleString() || 0}</p>
            </div>
            <div className="space-y-1">
              <p className="text-sm text-muted-foreground">Avg Utilization</p>
              <p className="text-2xl font-bold">
                {gpuData.length > 0 
                  ? (gpuData.reduce((sum, d) => sum + d.utilization, 0) / gpuData.length).toFixed(1)
                  : 0}%
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
