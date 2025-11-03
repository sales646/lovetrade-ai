import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { useDistributedTrainingStatus, usePBTHistory } from "@/lib/api/distributed-training";
import { Activity, Cpu, Zap, TrendingUp, Users, Layers } from "lucide-react";

export default function Strategies() {
  const { data: distStatus } = useDistributedTrainingStatus();
  const { data: pbtHistory } = usePBTHistory(20);

  return (
    <div className="space-y-6">
      {/* Distributed Training Status */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>Distributed RL Training</CardTitle>
              <CardDescription>8 GPUs • Population-Based Training • Transformer Policies</CardDescription>
            </div>
            {distStatus?.is_active && (
              <Badge variant="default" className="gap-2">
                <Activity className="h-3 w-3 animate-pulse" />
                Training Active
              </Badge>
            )}
          </div>
        </CardHeader>
        <CardContent>
          {distStatus ? (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="space-y-2">
                <div className="flex items-center gap-2 text-muted-foreground">
                  <Cpu className="h-4 w-4" />
                  <span className="text-sm">GPUs Active</span>
                </div>
                <p className="text-2xl font-bold">{distStatus.total_gpus}</p>
              </div>

              <div className="space-y-2">
                <div className="flex items-center gap-2 text-muted-foreground">
                  <Layers className="h-4 w-4" />
                  <span className="text-sm">Environments</span>
                </div>
                <p className="text-2xl font-bold">{distStatus.total_environments.toLocaleString()}</p>
              </div>

              <div className="space-y-2">
                <div className="flex items-center gap-2 text-muted-foreground">
                  <TrendingUp className="h-4 w-4" />
                  <span className="text-sm">Avg Reward</span>
                </div>
                <p className="text-2xl font-bold">{distStatus.performance.avg_reward.toFixed(2)}</p>
              </div>

              <div className="space-y-2">
                <div className="flex items-center gap-2 text-muted-foreground">
                  <Zap className="h-4 w-4" />
                  <span className="text-sm">Avg Loss</span>
                </div>
                <p className="text-2xl font-bold">{distStatus.performance.avg_loss.toFixed(4)}</p>
              </div>
            </div>
          ) : (
            <p className="text-muted-foreground">Loading distributed training status...</p>
          )}
        </CardContent>
      </Card>

      {/* PBT Status */}
      {distStatus?.pbt && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Users className="h-5 w-5" />
              Population-Based Training
            </CardTitle>
            <CardDescription>Auto-tuning hyperparameters during training</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="space-y-2">
                <p className="text-sm text-muted-foreground">Generation</p>
                <p className="text-2xl font-bold">{distStatus.pbt.generation}</p>
              </div>

              <div className="space-y-2">
                <p className="text-sm text-muted-foreground">Best Performance</p>
                <p className="text-2xl font-bold text-green-500">
                  {distStatus.pbt.best_performance.toFixed(2)}
                </p>
              </div>

              <div className="space-y-2">
                <p className="text-sm text-muted-foreground">Mean Performance</p>
                <p className="text-2xl font-bold">{distStatus.pbt.mean_performance.toFixed(2)}</p>
              </div>

              <div className="space-y-2">
                <p className="text-sm text-muted-foreground">Population</p>
                <p className="text-2xl font-bold">{distStatus.pbt.population_size}</p>
              </div>
            </div>

            {pbtHistory && pbtHistory.length > 0 && (
              <div className="mt-4 pt-4 border-t">
                <p className="text-sm font-medium mb-2">Recent PBT History</p>
                <div className="space-y-1">
                  {pbtHistory.slice(0, 5).map((gen: any) => (
                    <div key={gen.id} className="flex items-center justify-between text-sm">
                      <span className="text-muted-foreground">Gen {gen.generation}</span>
                      <span className="font-mono">
                        Best: {gen.best_performance?.toFixed(2)} | 
                        Mean: {gen.mean_performance?.toFixed(2)}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Recent Runs */}
      {distStatus?.recent_runs && distStatus.recent_runs.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Recent Training Runs</CardTitle>
            <CardDescription>Active and completed distributed training sessions</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {distStatus.recent_runs.map((run) => (
                <div key={run.id} className="flex items-center justify-between p-3 border rounded-lg">
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <Badge variant={run.status === 'running' ? 'default' : 'secondary'}>
                        {run.status}
                      </Badge>
                      <span className="text-sm font-mono">{run.id.slice(0, 8)}</span>
                    </div>
                    <p className="text-sm text-muted-foreground">
                      {run.config.world_size} GPUs × {run.config.envs_per_gpu} envs | 
                      {run.config.model_type} | 
                      {run.config.pbt_enabled ? 'PBT enabled' : 'No PBT'}
                    </p>
                  </div>
                  <div className="text-right text-sm text-muted-foreground">
                    {new Date(run.started_at).toLocaleString()}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* System Info */}
      <Card>
        <CardHeader>
          <CardTitle>System Capabilities</CardTitle>
          <CardDescription>Distributed training system information</CardDescription>
        </CardHeader>
        <CardContent>
          {distStatus?.system_info && (
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm">Distributed Training</span>
                <Badge variant={distStatus.system_info.distributed_available ? "default" : "secondary"}>
                  {distStatus.system_info.distributed_available ? "Available" : "Not Available"}
                </Badge>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm">Maximum GPUs</span>
                <span className="font-mono">{distStatus.system_info.max_gpus}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm">BF16 Precision</span>
                <Badge variant={distStatus.system_info.bf16_supported ? "default" : "secondary"}>
                  {distStatus.system_info.bf16_supported ? "Supported" : "Not Supported"}
                </Badge>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
