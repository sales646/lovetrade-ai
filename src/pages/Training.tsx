import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

export default function Training() {
  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Model Training</CardTitle>
          <CardDescription>Train and evaluate trading models</CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground">Training interface coming soon...</p>
        </CardContent>
      </Card>
    </div>
  );
}
