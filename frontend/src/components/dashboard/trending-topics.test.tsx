import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { TrendingTopics } from "./trending-topics";
import type { TrendingTopic } from "@/types/dashboard";

const mockTopics: TrendingTopic[] = [
  { topic: "Large Language Models", count: 89 },
  { topic: "Retrieval-Augmented Generation", count: 67 },
  { topic: "Diffusion Models", count: 54 },
  { topic: "Multimodal Learning", count: 48 },
];

describe("TrendingTopics", () => {
  it("renders all topic tags", () => {
    render(<TrendingTopics topics={mockTopics} />);

    expect(screen.getByText("Large Language Models")).toBeInTheDocument();
    expect(screen.getByText("Retrieval-Augmented Generation")).toBeInTheDocument();
    expect(screen.getByText("Diffusion Models")).toBeInTheDocument();
    expect(screen.getByText("Multimodal Learning")).toBeInTheDocument();
  });

  it("shows frequency counts", () => {
    render(<TrendingTopics topics={mockTopics} />);

    expect(screen.getByText("(89)")).toBeInTheDocument();
    expect(screen.getByText("(67)")).toBeInTheDocument();
  });

  it("shows skeleton when loading", () => {
    render(<TrendingTopics topics={[]} loading />);

    expect(screen.getByTestId("trending-topics-skeleton")).toBeInTheDocument();
  });

  it("shows empty state when no topics", () => {
    render(<TrendingTopics topics={[]} />);

    expect(screen.getByTestId("trending-topics-empty")).toBeInTheDocument();
    expect(screen.getByText("No trending topics")).toBeInTheDocument();
  });

  it("has trending-topics test id", () => {
    render(<TrendingTopics topics={mockTopics} />);
    expect(screen.getByTestId("trending-topics")).toBeInTheDocument();
  });

  it("applies larger font to higher-count topics", () => {
    const topics: TrendingTopic[] = [
      { topic: "High Count", count: 100 },
      { topic: "Low Count", count: 10 },
    ];
    render(<TrendingTopics topics={topics} />);

    const highCountTag = screen.getByText("High Count").closest("span");
    const lowCountTag = screen.getByText("Low Count").closest("span");

    expect(highCountTag?.className).toContain("text-base");
    expect(lowCountTag?.className).toContain("text-xs");
  });
});
