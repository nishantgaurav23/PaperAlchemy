import { HeroSection } from "@/components/landing/hero-section";
import { FeatureGrid } from "@/components/landing/feature-grid";
import { StatsCounter } from "@/components/landing/stats-counter";
import { UseCases } from "@/components/landing/use-cases";
import { LandingFooter } from "@/components/landing/landing-footer";
import { PageTransition } from "@/components/animations/page-transition";

export default function Home() {
  return (
    <PageTransition>
      <div className="flex flex-col">
        <HeroSection />
        <FeatureGrid />
        <StatsCounter />
        <UseCases />
        <LandingFooter />
      </div>
    </PageTransition>
  );
}
