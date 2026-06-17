from backend.Agents.summarizer import TechnicalTextSummarizer

if __name__ == "__main__":
    try:
        input = """
        The image depicts a single, large, glossy black plastic trash bag that has been filled and tied at the top with a knot. The bag is bulging, indicating it is full of discarded materials. Next to the filled bag is a neat stack of unused black plastic liners, suggesting a bulk supply of these bags. Th...
Image Description The image depicts a single, large, glossy black plastic trash bag that has been filled and tied at the top with a knot. The bag is bulging, indicating it is full of discarded materials. Next to the filled bag is a neat stack of unused black plastic liners, suggesting a bulk supply of these bags. The background is a stark, clean white, which isolates the subject and emphasizes the texture and reflective surface of the plastic. Relation to Waste, Pollution, and Sustainability This image serves as a powerful visual representation of the modern "throwaway culture." Here is how it relates to key environmental themes: 1. Waste Generation The filled bag represents the end-stage of a linear economy: Take Make Dispose. It symbolizes the volume of household or commercial waste generated daily. The stack of extra bags highlights the systemic reliance on single-use plastics to manage this waste, creating a cycle where the tool used to contain waste is itself a waste product. 2. Pollution Plastic Pollution: These bags are typically made from low-density polyethylene (LDPE), a petroleum-based plastic. Because they are designed for disposal, they often end up in landfills or leak into the environment. Microplastics: Plastic bags do not biodegrade; instead, they photodegrade, meaning they break down into smaller and smaller fragments called microplastics. These permeate soil and water systems, entering the food chain and harming wildlife. Chemical Leaching: As the contents of the bag decompose in a landfill, the plastic can leach chemicals into the groundwater, contributing to soil and water pollution. 3. Sustainability From a sustainability perspective, this image represents an "unsustainable" model. To move toward a more sustainable future, the items in this image would need to be replaced or reimagined through the following lenses: Reduction: Reducing the need for liners altogether through composting organic waste and recycling dry materials. Material Substitution: Replacing petroleum-based plastic bags with compostable or biodegradable alternatives made from cornstarch or other plant-based polymers. Circular Economy: Moving away from the "trash bag" model toward a circular system where products are designed for reuse or complete recovery, eliminating the need for a "waste bag" entirely.
        """

        summarizer = TechnicalTextSummarizer()

        output = summarizer.summarize(input)

        print("\nFINAL SUMMARY\n")
        print(output)

    except Exception as error:
        print(f"Processing failed: {error}")