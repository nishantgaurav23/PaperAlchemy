import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { HeroSection } from "./hero-section";
import { FeatureGrid } from "./feature-grid";
import { StatsCounter } from "./stats-counter";
import { UseCases } from "./use-cases";
import { LandingFooter } from "./landing-footer";

// Mock next/link
vi.mock("next/link", () => ({
  default: ({
    children,
    href,
    ...props
  }: {
    children: React.ReactNode;
    href: string;
    [key: string]: unknown;
  }) => (
    <a href={href} {...props}>
      {children}
    </a>
  ),
}));

// Mock IntersectionObserver
beforeEach(() => {
  class MockIntersectionObserver {
    callback: IntersectionObserverCallback;
    constructor(callback: IntersectionObserverCallback) {
      this.callback = callback;
    }
    observe() {
      // Immediately trigger with isIntersecting: true
      this.callback(
        [{ isIntersecting: true } as IntersectionObserverEntry],
        this as unknown as IntersectionObserver,
      );
    }
    unobserve() {}
    disconnect() {}
  }
  vi.stubGlobal("IntersectionObserver", MockIntersectionObserver);
});

describe("HeroSection", () => {
  it("renders headline and subtext", () => {
    render(<HeroSection />);
    expect(
      screen.getByRole("heading", { level: 1 }),
    ).toBeInTheDocument();
    expect(screen.getByText(/research papers/i)).toBeInTheDocument();
  });

  it("renders two CTA buttons linking to /chat and /search", () => {
    render(<HeroSection />);
    const chatLink = screen.getByRole("link", { name: /get started/i });
    const searchLink = screen.getByRole("link", { name: /explore papers/i });
    expect(chatLink).toHaveAttribute("href", "/chat");
    expect(searchLink).toHaveAttribute("href", "/search");
  });

  it("has animated gradient background element", () => {
    const { container } = render(<HeroSection />);
    const gradientEl = container.querySelector("[data-testid='mesh-gradient']");
    expect(gradientEl).toBeInTheDocument();
  });
});

describe("FeatureGrid", () => {
  it("renders exactly 6 feature cards", () => {
    render(<FeatureGrid />);
    const cards = screen.getAllByRole("article");
    expect(cards).toHaveLength(6);
  });

  it("each card has a title and description", () => {
    render(<FeatureGrid />);
    const cards = screen.getAllByRole("article");
    cards.forEach((card) => {
      const heading = card.querySelector("h3");
      const description = card.querySelector("p");
      expect(heading).toBeInTheDocument();
      expect(description).toBeInTheDocument();
    });
  });

  it("renders a section heading", () => {
    render(<FeatureGrid />);
    expect(
      screen.getByRole("heading", { name: /features/i }),
    ).toBeInTheDocument();
  });
});

describe("StatsCounter", () => {
  it("renders at least 3 stat items", () => {
    render(<StatsCounter />);
    const statItems = screen.getAllByTestId("stat-item");
    expect(statItems.length).toBeGreaterThanOrEqual(3);
  });

  it("each stat has a number and label", () => {
    render(<StatsCounter />);
    const statItems = screen.getAllByTestId("stat-item");
    statItems.forEach((item) => {
      const number = item.querySelector("[data-testid='stat-number']");
      const label = item.querySelector("[data-testid='stat-label']");
      expect(number).toBeInTheDocument();
      expect(label).toBeInTheDocument();
    });
  });
});

describe("UseCases", () => {
  it("renders 3 use case items", () => {
    render(<UseCases />);
    const items = screen.getAllByRole("article");
    expect(items).toHaveLength(3);
  });

  it("each use case has a title and description", () => {
    render(<UseCases />);
    const items = screen.getAllByRole("article");
    items.forEach((item) => {
      const heading = item.querySelector("h3");
      const description = item.querySelector("p");
      expect(heading).toBeInTheDocument();
      expect(description).toBeInTheDocument();
    });
  });

  it("renders a section heading", () => {
    render(<UseCases />);
    expect(
      screen.getByRole("heading", { name: /use cases/i }),
    ).toBeInTheDocument();
  });
});

describe("LandingFooter", () => {
  it("renders PaperAlchemy branding", () => {
    render(<LandingFooter />);
    expect(screen.getAllByText(/paperalchemy/i).length).toBeGreaterThanOrEqual(1);
  });

  it("renders navigation link sections", () => {
    render(<LandingFooter />);
    expect(screen.getByText(/product/i)).toBeInTheDocument();
    expect(screen.getByText(/resources/i)).toBeInTheDocument();
  });

  it("contains links to key pages", () => {
    render(<LandingFooter />);
    const links = screen.getAllByRole("link");
    expect(links.length).toBeGreaterThanOrEqual(3);
  });
});
