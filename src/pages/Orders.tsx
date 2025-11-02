import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

export default function Orders() {
  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Orders & Positions</CardTitle>
          <CardDescription>Manage your trading activity</CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground">Order ticket and positions coming soon...</p>
        </CardContent>
      </Card>
    </div>
  );
}
