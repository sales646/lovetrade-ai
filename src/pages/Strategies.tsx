import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

export default function Strategies() {
  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Trading Strategies</CardTitle>
          <CardDescription>Configure and manage your strategy ensemble</CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground">Strategy configuration coming soon...</p>
        </CardContent>
      </Card>
    </div>
  );
}
