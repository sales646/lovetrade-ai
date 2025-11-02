import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

export default function Logs() {
  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>System Logs</CardTitle>
          <CardDescription>Monitor system events and activities</CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground">Logs viewer coming soon...</p>
        </CardContent>
      </Card>
    </div>
  );
}
