import Link from "next/link";
import { Button } from "@/components/ui/button";
import { MessageSquare, Search } from "lucide-react";

export function HeroSection() {
  return (
    <section className="relative flex min-h-[70vh] items-center justify-center overflow-hidden px-4 py-24">
      {/* Animated mesh gradient background */}
      <div
        data-testid="mesh-gradient"
        className="pointer-events-none absolute inset-0 -z-10 animate-mesh-gradient opacity-30"
        style={{
          background:
            "radial-gradient(ellipse at 20% 50%, hsl(var(--primary) / 0.4) 0%, transparent 50%), " +
            "radial-gradient(ellipse at 80% 20%, hsl(270 80% 60% / 0.3) 0%, transparent 50%), " +
            "radial-gradient(ellipse at 50% 80%, hsl(200 80% 60% / 0.3) 0%, transparent 50%)",
          backgroundSize: "200% 200%",
        }}
      />

      <div className="mx-auto flex max-w-3xl flex-col items-center gap-8 text-center">
        <h1 className="text-4xl font-bold tracking-tight sm:text-5xl lg:text-6xl">
          Transform How You Read{" "}
          <span className="bg-gradient-to-r from-primary to-violet-500 bg-clip-text text-transparent">
            Research Papers
          </span>
        </h1>

        <p className="max-w-xl text-lg text-muted-foreground sm:text-xl">
          AI-powered research assistant that searches, analyzes, and synthesizes
          academic papers — with every answer backed by real citations.
        </p>

        <div className="flex flex-col gap-4 sm:flex-row">
          <Button asChild size="lg" className="gap-2">
            <Link href="/chat">
              <MessageSquare className="h-5 w-5" />
              Get Started
            </Link>
          </Button>
          <Button asChild variant="outline" size="lg" className="gap-2">
            <Link href="/search">
              <Search className="h-5 w-5" />
              Explore Papers
            </Link>
          </Button>
        </div>
      </div>
    </section>
  );
}
