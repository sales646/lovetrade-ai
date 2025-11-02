-- Allow service role to insert training metrics from GPU training
CREATE POLICY "Service role can insert training_metrics"
ON public.training_metrics
FOR INSERT
TO service_role
WITH CHECK (true);

-- Allow service role to update training metrics
CREATE POLICY "Service role can update training_metrics"
ON public.training_metrics
FOR UPDATE
TO service_role
USING (true)
WITH CHECK (true);